from text_preprocessing import normalize
NO_RULE_ERROR_CODE = -404


class NLGRule:
    def __init__(self, relation_set, rule_code):
        self.relation_set = relation_set
        self.rule_code = rule_code

    def prepare_exec_code(self, triplets):
        pass

    def exec_rule(self, triplets):
        result_dict = {}
        combined_script = self.prepare_exec_code(triplets)
        try:
            # Execute the combined script with a custom local namespace
            exec(combined_script, globals(), locals())
            # Get the updated output from the result_dict
            output = result_dict.get('output', '')
            return output, None
        except Exception as e:
            # Handle exceptions
            output = result_dict.get('output', '')
            return output, str(e)


class TemplateRule(NLGRule):
    def __init__(self, relation_set, rule_code):
        self.relation_set = relation_set
        self.template = rule_code

    def prepare_exec_code(self, triplets):
        pass

    def fill_template(self,triple):
        """
        Fills a template with the data from the triple
        """
        for item, placeholder in [
            (triple.subj, "<subject>"), 
            (triple.pred, "<predicate>"), 
            (triple.obj, "<object>")
        ]:
            template = self.template.replace(placeholder, normalize(item, 
                    remove_quotes=True, 
                    remove_parentheses=True))
        return template
    
    def exec_rule(self, triplets):
        if len(triplets)> 1 :
            print(f"ERROR: Template rule used for more triples than 1 ({self.relation_set}), {triplets}")
        out = [self.fill_template(triplet) for triplet in triplets ]
        return " ".join(out), None


class Program:
    def __init__(self):
        self.rules = {}

    def add_rule(self, rule):
        relationset_str = tuple(sorted(rule.relation_set))
        if relationset_str in self.rules:
            print(f"WARN: Replacing an existing rule for {rule.relation_set}")
        self.rules[relationset_str] = rule

    def exec(self, relations, triplets):
        relationset_str = tuple(sorted(relations))
        if relationset_str in self.rules:
            return self.rules[relationset_str].exec_rule(triplets)
        return NO_RULE_ERROR_CODE, f"No rule for a given combination {relations}"

    def process_input(self, relations, triplets):
        out, err = self.exec(relations, triplets)
        if out == NO_RULE_ERROR_CODE:
            known_relations = self.get_known_relations()
            is_in_domain = all(rel in known_relations for rel in relations)
            if not is_in_domain:
                return "OUT OF DOMAIN"
            else:
                result = []
                for rel, trip in self._make_split(relations, triplets):
                    result.append(self.process_input(rel,trip))
                return " ".join(result)
        return out

    def _make_split(self, relations, triplets):
        relations_set = set(relations)
        if len(relations_set) != len(relations):
            print(f"ERROR: {triplets}")
        available_rules = [set(r) for r in self.rules]
        result = [] 
        while len(relations_set) != 0:
            available_rules = [r for r in available_rules if r.issubset(relations_set)]
            if len(available_rules) == 0:
                print(f"{relations_set}: {available_rules}")
                break
            best_rule = max(available_rules, key=len)
            best_triplets = [t for t in triplets if t.pred in best_rule]
            result.append((best_rule,best_triplets))
            relations_set = relations_set - best_rule
        return result


        
    def has_rule(self, relations):
        relationset_str = tuple(sorted(relations))
        return relationset_str in self.rules

    def get_known_relations(self):
        relations = []
        for rule in self.rules.values():
            relations.extend(list(rule.relation_set))
        return set(relations)

    def write_program(self, output_dir, name):
        writer = ProgramWriter(output_dir, name)
        for rule in self.rules.values():
            writer.add_rule(rule.relation_set, rule.rule_code)
        writer.add_print_stmt()
        writer.write_program()

    def add_json_templates(self, templates_filename):

        import json
        with open(templates_filename) as f:
            templates = json.load(f)
        for relation in templates:
            relationName = normalize(relation)
            relationset_str = tuple(sorted([relationName]))
            if relationset_str not in self.rules:
                rule = TemplateRule(set([relationName]), templates[relation][0])
                self.add_rule(rule)

class ProgramWriter:
    def __init__(self, output_dir, name):
        self.name = name
        self.output_dir = output_dir
        self.state = "initial"

        self.program = ""
        self.__add_header()

    def __add_header(self):
        # read header from file and add to program
        header_file = self.output_dir / "header.py"
        with open(header_file, "r", encoding="utf-8") as f:
            header = f.read()
            self.program += header

    def add_rule(self, relation_set, rule):
        if_statement = f'''
    if relations == {relation_set}:'''
        self.program += if_statement
        rule = self.adjust_indentation(rule, 8)
        self.program += rule
        self.state = "pending if stmts"

    def adjust_indentation(self, rule, indent_level):
        # Split the rule string into lines
        rule_lines = rule.split("\n")

        existing_indentation = 0
        # Find the first non-empty line and determine its indentation
        for line in rule_lines:
            if line.strip():  # Check if the line is not empty
                existing_indentation = len(line) - len(line.lstrip())
                break

        # Add additional spaces to each line based on the existing indentation
        indented_rule = "\n".join(" " * indent_level + line[existing_indentation:] for line in rule_lines)

        # Add the indented rule to the program
        return indented_rule

    def add_print_stmt(self):
        print_stmt = '''
    print(output)'''
        self.program += print_stmt
        self.state = "initial"

    def write_program(self):
        output_file = self.output_dir / f"{self.name}.py"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(self.program)
