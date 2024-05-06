import ollama


class LMPoller:
    def __init__(self):
        self.prompt_template = self.__get_prompt_template()
        self.errors_query_template = self.__get_errors_query_template()
        self.wrong_output_query_template = self.__get_wrong_output_query_template()

        self.last_prompt = None
        self.last_response = None

        self.last_triplets = None
        self.last_reference_text = None
        self.last_relation_set = None

    def __get_prompt_template(self):
        with open('../res/prompt_templates/template1.txt', 'r') as file:
            return file.read()

    def __get_errors_query_template(self):
        # TODO: define a template for fixing prompt due to errors in generated code response
        with open('../res/prompt_templates/error_template.txt', 'r') as file:
            return file.read()

    def __get_wrong_output_query_template(self):
        # TODO: define a template for fixing prompt due to wrong output
        with open('../res/prompt_templates/wrong_output_template.txt', 'r') as file:
            return file.read()

    def query_lm(self, triplets, reference_text, relation_set):
        self.last_triplets = triplets
        self.last_reference_text = reference_text
        self.last_relation_set = relation_set

        self.last_prompt = self.prompt_template.format(triplets, reference_text, relation_set)
        self.last_response = ollama.chat(model='llama3:70b', messages=[
            {
                'role': 'user',
                'content': self.last_prompt,
            },
        ])['message']['content']
        return self.last_response

    def fix_query(self, incorrect_output, errors):
        if errors:
            prompt = self.errors_query_template.format(
                self.last_reference_text, errors
            )
        else:
            prompt = self.wrong_output_query_template.format(
                self.last_reference_text, incorrect_output
            )

        self.last_response = ollama.chat(model='llama3:70b', messages=[
            {
                'role': 'user',
                'content': self.last_prompt,
            },
            {
                'role': 'assistant',
                'content': self.last_response
            },
            {
                'role': 'user',
                'content': prompt
            }
        ])['message']['content']
        return self.last_response
