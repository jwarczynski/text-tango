import os
from argparse import ArgumentParser
from pathlib import Path

import ollama

from typing import List, Any
from tqdm import tqdm

from utils import get_logger
from samplesfs.fs import SamplesFS


class ResponsesHandler:
    def __init__(self, logger=None):
        self.logger = logger

    def extract_sample(self, response, relations) -> dict[str, str] | None:
        raise NotImplementedError

    def _extract_from_tags(self, response, start_tag="<sample>", end_tag="</sample>"):
        try:
            sample = response.strip().split(start_tag)[-1].split(end_tag)[0]
            return sample

        except Exception as e:
            self.logger.error(f'Failed to extract sample from response: {response} due to {e}') if self.logger else None
            return None


class StringResponseHandler(ResponsesHandler):
    def __init__(self, logger=None):
        super().__init__(logger)

    def extract_sample(self, response, relations) -> dict[str, str] | None:
        sample = self._extract_from_tags(response)
        triples = sample.split('"in: "')[-1].split('"out: "')[0].strip()
        output = sample.split("out: ")[-1].strip()

        return {
            'in': triples,
            'out': output,
        }


class ExecResponseHandler(ResponsesHandler):
    def __init__(self, logger=None):
        super().__init__(logger)

    def extract_sample(self, response, relations) -> dict[str, Any] | None:
        response = self._extract_from_tags(response)
        # self.logger.info(f'Extracted response: {response}') if self.logger else None
        sample_json = '{' + response + '}'
        # self.logger.info(f'Parsed json response: {sample_json}') if self.logger else None
        sample = eval(sample_json)
        # self.logger.info(f'eval json response: {sample}') if self.logger else None
        sample["in"] = [tuple(x.lower() for x in triple) for triple in sample["in"]]
        # self.logger.info(f'Lowercased triples: {sample["in"]}') if self.logger else None
        sample["in"] = self.__reorrder_predicates(sample["in"], relations)
        # self.logger.info(f'Reordered triples: {sample["in"]}') if self.logger else None
        return sample

    def __reorrder_predicates(self, triplets: List[tuple[str]], relations: List[str]):
        # self.logger.info(f'Reordering triples: {triplets}. Relations: {relations}') if self.logger else None
        for rel in relations:
            for i, triple in enumerate(triplets):
                if rel in triple:
                    triplets[i] = self.__reorder_triplet_elements(triple, rel)

        return triplets

    def __reorder_triplet_elements(self, triple, rel):
        # self.logger.info(f'Reordering triple: {triple}. Relation: {rel}') if self.logger else None
        if triple[1] == rel:
            return triple
        triple = list(triple)
        if triple[0] == rel:
            triple[0], triple[1] = triple[1], triple[0]
        elif triple[2] == rel:
            triple[1], triple[2] = triple[2], triple[1]
        triple = tuple(triple)

        return triple


class Augmentation:
    def __init__(
            self, prompt_template: str,
            responses_handler: ResponsesHandler,
            logger,
            model_name='llama3:70b',
            checkpoint_interval=100,
    ):

        self.responses_handler = responses_handler
        self.__model = model_name
        self.prompt_template = prompt_template
        self.logger = logger
        self.__checkpoint_interval = checkpoint_interval

        self.__temperature = 0
        self.samples = []
        self.not_augmented = []

        self.__log_info(
            f'Initialized augmentation with model {model_name}.'
            f'Checkpoint interval: {checkpoint_interval}.'
            f'Responses handler: {responses_handler.__class__.__name__}'
        )

    def augment(self, samples_fs: SamplesFS,):
        relations = samples_fs.load_samples()
        for relation in tqdm(relations):
            self.__generate_sample(relation)
            if len(self.samples) % self.__checkpoint_interval == 0:
                self._samples_fs.write_samples(self.samples, self.not_augmented)
                self.samples = []
                self.not_augmented = []

    def __generate_sample(self, relation):
        prompt = self.prompt_template.format(relations=relation)
        self.__tries = 0
        is_valid = False
        while not is_valid and self.__tries < 10:
            response = self.__query_lm(prompt)
            candidate = self.__extract_sample(response, relation)
            self.__tries += 1
            is_valid = self.__is_valid(candidate, relation)

        if is_valid:
            self.samples.append(candidate)
            self.logger.info(f'Generated sample for relation {relation}') if self.logger else None
        else:
            self.logger.warning(f'Could not generate sample for relation {relation}') if self.logger else None

    def __query_lm(self, prompt) -> str:
        return ollama.chat(model=self.__model, messages=[
            {
                'role': 'user',
                'content': prompt,
                "options": {
                    # "seed": 101,
                    "temperature": self.__tries * 0.1
                }
            },

        ])['message']['content']

    def __extract_sample(self, response, relation) -> dict[str, str] | None:
        try:
            return self.responses_handler.extract_sample(response, relation)
        except Exception as e:
            self.logger.error(f'Failed to extract sample from response: {response} due to {e}') if self.logger else None
            return None

    def __is_valid(self, candidate, relation_set) -> bool:
        if candidate is None:
            self.logger.warning('Candidate is None')
            return False
        if self.__not_contains_all_relation(candidate['in'], relation_set):
            return False
        if self.__triples_invalid(candidate['in'], relation_set):
            return False

        return True

    def __not_contains_all_relation(self, text, relation_set):
        # self.logger.info(f'Checking if {text} contains all relations {relation_set}') if self.logger else None
        if type(text) == tuple:
            predicates = [triple[1] for triple in text]
            for rel in relation_set:
                if rel not in predicates:
                    self.logger.warning(f'Relation {rel} not in {text}')
                    return True
        elif type(text) == str:
            for relation in relation_set:
                if relation not in text:
                    self.logger.warning(f'Relation {relation} not in {text}')
                    return True
        return False

    def __triples_invalid(self, triples, relation_set):
        return False

    def __log_info(self, message):
        self.logger.info(f'[AUGMENTATION] {message}') if self.logger else None


def read_relations(file_path: str) -> List[set]:
    with open(file_path, 'r') as file:
        data = file.readlines()
        data = [set(line.strip().split(';')) for line in data]
    return data


def read_prompt_template(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()


def parse_args():
    script_dir = os.path.dirname(__file__)
    default_log_dir = Path(script_dir).parent / 'logs'
    default_output_dir = Path(script_dir).parent / 'out' / 'augmentation'
    default_relations_file = Path(script_dir).parent / 'res' / 'augmentation_relations.csv'
    default_prompt_template_path = Path(script_dir).parent / 'res' / 'prompt_templates' / 'generate_training_sample.txt'
    default_checkpoint_dir = Path(script_dir).parent / 'checkpoints'

    parser = ArgumentParser()
    parser.add_argument("--log-dir", "-ld", default=default_log_dir, type=str, help="log directory")
    parser.add_argument("--output-dir", "-od", default=default_output_dir, type=str, help="output file")
    parser.add_argument("--relations-file", "-rf", default=default_relations_file, type=str, help="relations file")
    parser.add_argument("--prompt-template", "-pt", default=default_prompt_template_path, type=str,
                        help="prompt template file")
    parser.add_argument("--checkpoint-dir", "-cd", default=default_checkpoint_dir, type=str, help="checkpoint directory")

    parser.add_argument("--responses-handler", "-rh", default='exec', type=str, help="responses handler")
    parser.add_argument("--model", "-m", default='llama3:70b', type=str, help="language model")
    parser.add_argument("--checkpoint-interval", "-ci", default=100, type=int, help="checkpoint interval")

    return parser.parse_args()


def get_response_handler(type: str, logger) -> ResponsesHandler:
    if type == 'string':
        return StringResponseHandler(logger)
    elif type == 'exec':
        return ExecResponseHandler(logger)
    else:
        raise ValueError(f'Unknown responses handler: {args.responses_handler}')


if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(args.log_dir, 'augmentation', stream=True, time_in_filename=True)

    logger.info(f'Reading relations from {args.relations_file}')
    relations = read_relations(args.relations_file)
    logger.info(f'Read {len(relations)} relations')

    logger.info(f'Reading prompt template from {args.prompt_template}')
    prompt_template = read_prompt_template(args.prompt_template)
    logger.info(f'Read prompt template')

    responses_handler = get_response_handler(args.responses_handler, logger)
    augmentation = Augmentation(prompt_template, responses_handler=responses_handler, model_name=args.model,
                                logger=logger, checkpoint_interval=args.checkpoint_interval)

    fs = SamplesFS(args.checkpoint_dir, args.relations_file, args.output_dir, logger)
    augmentation.augment(fs)

    logger.info(f'Finished augmentation. Generated {len(augmentation.samples)} samples')
