class Program:
    def __init__(self, name):
        self.name = name
        self.state = "initial"

        self.program=""
        self.__add_header()

    def __add_header(self):
        # read header from file and add to program
        with open("header.py", "r", encoding="utf-8") as f:
            header = f.read()
            self.program += header

    def add_rule_if_stmt(self, relation_set):
        if_statement = f'''
    if relations == {relation_set}:'''

        self.program += if_statement
        self.state = "pending rule stmt"

    def add_rule(self, rule):
        if self.state == "pending rule stmt":
            #for each line in rule add 4 spaces
            rule = rule.replace("\n", "\n    ")
            self.program += rule
            self.state = "pending if stmts"
        else:
            raise Exception("Rule statement must be preceded by a top-level if statement")

    def add_print_stmt(self):
        print_stmt = '''
    print(output)'''
        self.program += print_stmt
        self.state = "initial"

    def write_program(self):
        with open(f"{self.name}.py", "w", encoding="utf-8") as f:
            f.write(self.program)
