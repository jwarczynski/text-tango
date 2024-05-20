import os.path
import time
from collections import namedtuple
from pathlib import Path
from typing import List

import ollama

from lm_response_evaluator import SimpleNLGRule, extract_code, get_response_similarity
from program import Program
from logging import getLogger

from text_preprocessing import extract_triplets
from evaluate_program import RDFTriple, WebNLG

logger = getLogger('trainer')


class ProgramTrainer:

    def __init__(self):
        self.program = Program()

    def train(self, data_dict):
        #interate over data, call construct_rule
        #run enumerate_uknown_combinations and construct_new_combination
        pass

    def construct_rule(self, triplets, reference, sample_id):
        pass

    def enumerate_unknown_combinations(self):
        # perform clustering of relations
        # output what relation combinatinos should be covered by the rules
        pass

    def construct_new_combination(self, relations):
        # generate artificial reference text
        # run construct_rule()
        pass


TemplateTuple = namedtuple("TemplateTuple", ["first_query", "fix_query"])
ChatTuple = namedtuple("ChatTuple", ["role", "content"])


class LMResponseWriter:
    def __init__(self, raw_responses_dir, code_responses_dir, create_dirs=True):
        self.raw_responses_dir = Path(raw_responses_dir)
        self.code_responses_dir = Path(code_responses_dir)

        if create_dirs:
            self._create_dirs_if_not_exist()
        else:
            self.check_dirs_exist()

        self.raw_response_template = "QUERY_NUM: {query_number} - TIME: {time} - DURATION: {duration}s:\n{content}\n\n"
        self.code_responses_template = "QUERY_NUM: {query_number} - TIME: {time} - DURATION: {duration}s: {content}\n\n"

        self.chat_id = None
        self.__query_number = 0

    def new_chat(self, chat_id: str):
        self.chat_id = chat_id
        self.__query_number = 0

    def write_raw_response(self, response: str, duration: int):
        self.__write(
            response,
            duration,
            self.raw_response_template,
            self.raw_responses_dir / f"{self.chat_id}.raw"
        )

    def write_code(self, code: str, duration: int):
        self.__write(
            code,
            duration,
            self.code_responses_template,
            self.code_responses_dir / f"{self.chat_id}.code"
        )

    def __write(self, response: str, duration: int, template: str, path: Path):
        response = template.format(
            query_number=self.__query_number,
            time=time.strftime('%Y-%m-%d_%H-%M-%S'),
            duration=duration / 1_000_000_000,
            content=response
        )
        self.__query_number += 1

        with open(path, 'w') as f:
            f.write(response)

    def _create_dirs_if_not_exist(self):
        self.raw_responses_dir.mkdir(parents=True, exist_ok=True)
        self.code_responses_dir.mkdir(parents=True, exist_ok=True)

    def check_dirs_exist(self):
        if not self.raw_responses_dir.exists():
            raise FileNotFoundError(f"Directory {self.raw_responses_dir} does not exist.")
        if not self.code_responses_dir.exists():
            raise FileNotFoundError(f"Directory {self.code_responses_dir} does not exist.")


class RuleJudge:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def is_ok(self, rule_result, sample_out):
        return get_response_similarity(rule_result, sample_out) > self.threshold


class LanguageModel:
    def __init__(self, model_name="llama3:70b"):
        self.model_name = model_name

    def query(self, messages, temperature=0.7, seed=None):
        options = {
            "temperature": temperature,
            "seed": seed
        }

        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options=options,
            stream=False
        )

        return response['message']['content'], response['total_duration']


class Prompter:
    def __init__(self, lm: LanguageModel, templates: TemplateTuple, lm_response_writer: LMResponseWriter):
        self._lm = lm
        self.templates = templates
        self.lm_response_writer = lm_response_writer

        self._chat = []

    def new_chat(self, chat_id):
        self._chat = []
        self.lm_response_writer.new_chat(chat_id)

    def ask_for_code(self, triplets, reference) -> str:
        predicates = set([triplet.pred for triplet in triplets])
        prompt = self.templates.first_query.format(triplets, reference, predicates)
        return self._query_extract_save(prompt, triplets, reference, temperature=0, seed=None)

    def _query_extract_save(self, prompt, triplets, reference, temperature=0.7, seed=None):
        messages = self.__create_messages(prompt)

        response, duration = self._lm.query(messages, temperature=0, seed=None)
        self.lm_response_writer.write_raw_response(response, duration)

        code = self.__extract_code(response, triplets)
        self.lm_response_writer.write_code(code, duration)

        return code

    def __create_messages(self, prompt):
        self._chat.append(ChatTuple("user", prompt))
        messages = []
        for message in self._chat:
            msg = {
                "role": message.role,
                "content": message.content
            }
            messages.append(msg)

        return messages

    def __extract_code(self, response, relations):
        return extract_code(response, relations)

    def fix_code(self, reference, triplets, rule_result, ):
        prompt = self.templates.fix_query.format(reference, rule_result)
        temp = len(self._chat) / 10 / 2
        return self._query_extract_save(prompt, triplets, reference, temperature=temp, seed=None)


class SimpleProgramTrainer(ProgramTrainer):
    def __init__(self, lm: LanguageModel, templates: TemplateTuple, lm_response_writer: LMResponseWriter,
                 prompter: Prompter, judge: RuleJudge, max_fix_prompts=5):
        super().__init__()
        self.lm = lm
        self.templates = templates
        self.lm_response_writer = lm_response_writer
        self.prompter = prompter
        self._judge = judge
        self.max_fix_prompts = max_fix_prompts

    def train(self, dataset):
        for sample in dataset:
            predicates = set([triplet.pred for triplet in sample.data])
            if not self.program.has_rule(predicates):
                self.construct_rule(sample.data, sample.refs[0], sample.entry_id)

    def construct_rule(self, triplets: List[RDFTriple], reference: str, sample_id: str):
        def execute_rule(triplets, code):
            rule = SimpleNLGRule(triplets, code)
            output, errors = rule.exec_rule(triplets)
            if errors is not None:
                return errors, False
            return output, self._judge.is_ok(output, reference)

        self.prompter.new_chat(sample_id)
        fix_query_count = 0

        code = self.prompter.ask_for_code(triplets, reference)
        rule_result, is_rule_ok = execute_rule(triplets, code)

        while not is_rule_ok and fix_query_count < self.max_fix_prompts:
            code = self.prompter.fix_code(reference, triplets, rule_result)
            rule_result, is_rule_ok = execute_rule(triplets, code)
            fix_query_count += 1

        if is_rule_ok:
            self.program.add_rule(SimpleNLGRule(set([triplet.pred for triplet in triplets]), code))
        else:
            logger.error(
                f'[ObjectProgramTrainer] '
                f'Failed to generate rule for {triplets} after {fix_query_count} attempts. Skipping...'
            )


if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    lm_raw_responses_dir = Path(script_path) / '..' / 'out' / 'train' / 'lm_raw_responses'
    lm_code_responses_dir = Path(script_path) / '..' / 'out' / 'train' / 'lm_code_responses'
    first_query_template_path = Path(script_path) / '..' / 'res' / 'prompt_templates' / 'template1.txt'
    fix_query_template_path = Path(script_path) / '..' / 'res' / 'prompt_templates' / 'wrong_output_template.txt'
    output_program_dir = Path(script_path) / '..' / 'out' / 'train'
    output_program_name = 'rule_program'

    with open(first_query_template_path, 'r') as f:
        first_query_template = f.read()
    with open(fix_query_template_path, 'r') as f:
        fix_query_template = f.read()

    templates = TemplateTuple(first_query=first_query_template, fix_query=fix_query_template)
    lm_response_writer = LMResponseWriter(lm_raw_responses_dir, lm_code_responses_dir, create_dirs=True)
    lm = LanguageModel()
    prompter = Prompter(lm, templates, lm_response_writer)
    judge = RuleJudge()

    trainer = SimpleProgramTrainer(lm, templates, lm_response_writer, prompter, judge, 2)
    dataset = WebNLG()
    dataset.load(['train'])
    trainer.train(dataset.data)
    trainer.program.write_program(output_program_dir, output_program_name)
