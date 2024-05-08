import logging

from dataset import WebNLGDataset
from tqdm import tqdm
from pathlib import Path
import time

from text_preprocessing import extract_triplets
from lm_response_evaluator import extract_code, evaluate_response, get_response_similarity
from lm_poller import LMPoller
from program import Program
from dataset import WebNLGDataset


def train(dataset: WebNLGDataset, program:Program, lm_poller: LMPoller, max_fix_query=3, log_directory='../logs', subset_size=3):
    if subset_size is not None:
        dataset = dataset[:subset_size]

    logger = get_logger(log_directory, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_train.log", stream=True)
    for i, (key, value) in enumerate(tqdm(dataset)):
        sample_response_logger = get_logger(lm_poller.responses_dir, 'response', i)
        sample_exctracted_code_logger = get_logger(lm_poller.responses_dir, 'exctracted_code', i)

        relations = set(key)
        sample = value[0]
        sample_X = sample['in']
        reference_text = sample['out']
        triplets = extract_triplets(sample_X)

        response = lm_poller.query_lm(triplets, reference_text, relations)
        log_data = {
            'response': response,
            'metadata': {
                'fix_query_count': 0,
            }
        }
        sample_response_logger.info(log_data)

        exctracted_code = extract_code(response, relations)

        output, errors = evaluate_response(triplets, exctracted_code, reference_text, relations)
        fix_query_count = 0
        similarity = get_response_similarity(output, reference_text)

        # Log response and metadata to JSON file
        log_data = {
            'response': response,
            'metadata': {
                'fix_query_count': fix_query_count,
                'errors': errors,
                'similarity': similarity,
                'exctracted_code': exctracted_code,
                'output': output,
            }
        }

        sample_response_logger.info(log_data)
        sample_exctracted_code_logger.info(exctracted_code)

        while ((errors is not None or get_response_similarity(output, reference_text) < 0.5) and
               fix_query_count < max_fix_query):
            response = lm_poller.fix_query(output, errors)
            exctracted_code = extract_code(response, relations)
            output, errors = evaluate_response(triplets, exctracted_code, reference_text, relations)
            fix_query_count += 1
            log_data = {
                'response': response,
                'metadata': {
                    'fix_query_count': fix_query_count,
                    'errors': errors,
                    'similarity': get_response_similarity(output, reference_text),
                    'exctracted_code': exctracted_code,
                    'output': output,
                }
            }
            sample_response_logger.info(log_data)
            sample_exctracted_code_logger.info(exctracted_code)

        if fix_query_count < max_fix_query:
            program.add_rule(relations, exctracted_code)
        else:
            logger.error(f'Failed to generate rule for {key} after {max_fix_query} attempts. Skipping...')

    program.add_print_stmt()
    program.write_program()


def get_logger(directory, filename, sample_id=None, stream=False):
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True)

    file_path = directory / f'{filename}_{sample_id}.log'
    logger = logging.getLogger(f'{filename}_{sample_id}')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.handlers = [
        logging.FileHandler(file_path),
    ]

    if stream:
        logger.handlers.append(logging.StreamHandler())

    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger
