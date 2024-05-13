from argparse import ArgumentParser
from pathlib import Path
import time
import os
import pprint

from dataset import WebNLGDataset
from program import ProgramWriter
from lm_poller import LMPoller
from train import train
from utils import get_logger


def main():
    args = parse_args()
    logger = get_logger(args.log_dir, 'webnlg', stream=True)

    dataset = WebNLGDataset(data_dir=args.dataset_dir, split=args.split,
                            samples_per_relation_set=args.samples_per_relation_set)
    logger.info(f'Loaded {len(dataset)} samples')

    program = ProgramWriter(output_dir=Path(args.output_dir), name=args.program_name)
    logger.info(f'Initialized program. Output directory: {args.output_dir}')

    responses_dir = Path(args.lm_responses_dir) / args.model / f'{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    lm_poller = LMPoller(model_name=args.model, responses_dir=responses_dir)
    logger.info(f'Initialized language model poller. Responses directory: {responses_dir}')

    train(dataset, program, lm_poller, max_fix_query=args.max_lm_fix_queries, log_directory=args.log_dir)


def parse_args():
    parser = ArgumentParser()
    add_directory_arguments(parser)
    add_training_arguments(parser)
    args = parser.parse_args()

    pprint.pprint(vars(args))
    return args


def add_directory_arguments(parser):
    script_dir = os.path.dirname(__file__)
    default_dataset_dir = Path(script_dir).parent / 'res' / 'webnlg'
    default_log_dir = Path(script_dir).parent / 'logs'
    default_output_dir = Path(script_dir).parent / 'out'
    default_lm_responses_dir = Path(script_dir).parent / 'res' / 'lm_responses'

    directory_group = parser.add_argument_group('Directories')
    directory_group.add_argument("--dataset-dir", "-dd", default=default_dataset_dir, type=str, help="dataset directory")
    directory_group.add_argument("--log-dir", "-ld", default=default_log_dir, type=str, help="log directory")
    directory_group.add_argument("--output-dir", "-od", default=default_output_dir, type=str, help="output directory")
    directory_group.add_argument("--lm-responses-dir", "-lrd", default=default_lm_responses_dir,
                                 type=str, help="language model responses directory")

    directory_group.add_argument("--program-name", "-pn", default='program', type=str, help="generated program name")


def add_training_arguments(parser):
    training_group = parser.add_argument_group('Training')
    training_group.add_argument("--samples-per-relation-set", "-spr", default=1, type=int, help="number of samples per relation set")
    training_group.add_argument("--split", "-s", default='train', type=str, help="dataset split")
    training_group.add_argument("--model", "-m", default='llama3:70b', type=str, help="language model")
    training_group.add_argument("--max-lm-fix-queries", "-mlfq", default=3, type=int, help="maximum number of language model fix queries")


if __name__ == '__main__':
    main()
