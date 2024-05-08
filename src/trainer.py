from lm_poller import LMPoller
from lm_response_evaluator import SimpleNLGRule, extract_code, get_response_similarity
from program import Program
from logging import getLogger

from text_preprocessing import extract_triplets

logger = getLogger('trainer')

class ProgramTrainer:

    def __init__(self):
        self.program = Program()
        

    def train(self, data_dict):
        #interate over data, call construct_rule
        #run enumerate_uknown_combinations and construct_new_combination
        pass

    def construct_rule(self, triplets, reference):
        pass

    def enumerate_uknown_combinations(self):
        # perform clustering of relations
        # output what relation combinatinos should be covered by the rules
        pass

    def construct_new_combination(self, relations):
        # generate artificial reference text
        # run construct_rule()
        pass


#Copied from older JW's implementation
class SimpleProgramTrainer(ProgramTrainer):

    def __init__(self):
        super().__init__()
        self.lm = LMPoller()
        self.MAX_LLM_FIX_QUYERY = 5

    def train(self, data_dict):
        for i, key in enumerate(data_dict):
            relations = set(key)
            #get sample for the key
            sample = data_dict[key][0]
            sample_X = sample['in']
            reference_text = sample['out']
            triplets = extract_triplets(sample_X)
            
            # print(f'key: {key}, triplets: {triplets}, relations: {relations}')
            
            response = self.lm.query_lm(triplets, reference_text, relations)
            with open(f'../res/lama_responses/response_{i}.txt', 'w') as f:
                f.write(response)
            exctracted_code = extract_code(response, relations)
            rule = SimpleNLGRule(relations, exctracted_code)
            with open(f'../res/lama_responses/code_{i}.py', 'w') as f:
                f.write(exctracted_code)
            
            output, errors = rule.exec_rule(triplets) #evaluate_response(triplets ,exctracted_code, reference_text, relations)
            print(f'Errors: {errors}')
            fix_query_count = 0
            print(f'similarity: {get_response_similarity(output, reference_text)}')
            print(f'output: {output}\nreference: {reference_text}\n\n')
            while (errors is not None or get_response_similarity(output, reference_text) < 0.5) and fix_query_count < self.MAX_LLM_FIX_QUYERY:
                response = self.lm.fix_query(output, errors)
                exctracted_code = extract_code(response, relations)
                rule = SimpleNLGRule(relations, exctracted_code)
                output, errors = rule.exec_rule(triplets) # evaluate_response(triplets, exctracted_code, reference_text)
                fix_query_count += 1
                
            if fix_query_count < self.MAX_LLM_FIX_QUYERY:
                self.program.add_rule(rule)
            else:
                logger.error(f'Failed to generate rule for {key} after {self.MAX_LLM_FIX_QUYERY} attempts. Skipping...')
                
