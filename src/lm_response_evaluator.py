import difflib


def get_response_similarity(response, reference_text):
    return difflib.SequenceMatcher(None, response, reference_text).ratio()


def extract_code(response):
    # check if <code> tag is present in the response
    if '<code>' not in response:
        #try to extract code from ```
        if '```' in response:
            code = response.split('```')[1]
        else:
            # TODO: handle this case
            return None
    else:
        # Extract the code from the <code> tag
        code = response.split('<code>')[1].split('</code>')[0]

    # remove `print(output) from the code
    code = code.replace("print(output)", "")

    return code


def evaluate_response(triplets, code, reference_text):
    # Combine triplets, code, and reference text into a single Python script
    # remove indents from each line if the code in first line is indented

    code = remove_indents(code)

    combined_script = f"""
# Triplets
triplets = {triplets}
# Initialize output variable
output = ""
# Code
{code}
# Return the output
result_dict['output'] = output
"""
    with open('../res/combined_scripts/combined_script.py', 'w') as f:
        f.write(combined_script)

    result_dict = {}
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


def remove_indents(code):
    # Remove leading empty lines
    code_lines = code.split("\n")
    while code_lines and not code_lines[0].strip():
        code_lines.pop(0)

    # Determine the indentation of the first non-empty line
    if code_lines:
        indent = len(code_lines[0]) - len(code_lines[0].lstrip())
    else:
        indent = 0

    # Remove leading indentation from each line
    if indent > 0:
        code_lines = [line[indent:] for line in code_lines]

    return "\n".join(code_lines)
