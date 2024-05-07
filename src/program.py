class Program:
    def __init__(self, name):
        self.name = name
        self.state = "initial"

        self.program=""
        self.__add_header()

    def __add_header(self):
        # read header from file and add to program
        with open("../out/header.py", "r", encoding="utf-8") as f:
            header = f.read()
            self.program += header

    def add_rule_if_stmt(self, relation_set):
        if_statement = f'''
    if relations == {relation_set}:'''

        self.program += if_statement
        self.state = "pending rule stmt"

    def add_rule(self, rule):
        if self.state == "pending rule stmt":
            #for each line in rule add 8 spaces
            rule = self.adjust_indentation(rule, 8)
            self.program += rule
            self.state = "pending if stmts"
        else:
            raise Exception("Rule statement must be preceded by a top-level if statement")

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
        with open(f"{self.name}.py", "w", encoding="utf-8") as f:
            f.write(self.program)
