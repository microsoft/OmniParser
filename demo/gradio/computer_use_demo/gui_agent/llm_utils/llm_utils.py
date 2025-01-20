import os
import re
import ast
import base64


def is_image_path(text):
    # Checking if the input text ends with typical image file extensions
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")
    if text.endswith(image_extensions):
        return True
    else:
        return False


def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_url_or_filepath(input_string):
    # Check if input_string is a URL
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    if url_pattern.match(input_string):
        return "URL"

    # Check if input_string is a file path
    file_path = os.path.abspath(input_string)
    if os.path.exists(file_path):
        return "File path"

    return "Invalid"


def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string


def parse_input(code):
    """Use AST to parse the input string and extract the function name, arguments, and keyword arguments."""

    def get_target_names(target):
        """Recursively get all variable names from the assignment target."""
        if isinstance(target, ast.Name):
            return [target.id]
        elif isinstance(target, ast.Tuple):
            names = []
            for elt in target.elts:
                names.extend(get_target_names(elt))
            return names
        return []

    def extract_value(node):
        """提取 AST 节点的实际值"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            # TODO: a better way to handle variables
            raise ValueError(
                f"Arguments should be a Constant, got a variable {node.id} instead."
            )
        # 添加其他需要处理的 AST 节点类型
        return None

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = []
                for t in node.targets:
                    targets.extend(get_target_names(t))
                if isinstance(node.value, ast.Call):
                    func_name = node.value.func.id
                    args = [ast.dump(arg) for arg in node.value.args]
                    kwargs = {
                        kw.arg: extract_value(kw.value) for kw in node.value.keywords
                    }
                    print(f"Input: {code.strip()}")
                    print(f"Output Variables: {targets}")
                    print(f"Function Name: {func_name}")
                    print(f"Arguments: {args}")
                    print(f"Keyword Arguments: {kwargs}")
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                targets = []
                func_name = extract_value(node.value.func)
                args = [extract_value(arg) for arg in node.value.args]
                kwargs = {kw.arg: extract_value(kw.value) for kw in node.value.keywords}

    except SyntaxError:
        print(f"Input: {code.strip()}")
        print("No match found")

    return targets, func_name, args, kwargs


if __name__ == "__main__":
    import json
    s='{"Reasoning": "The Docker icon has been successfully clicked, and the Docker application should now be opening. No further actions are required.", "Next Action": None}'
    json_str = json.loads(s)
    print(json_str)