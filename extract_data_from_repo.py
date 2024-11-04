import os
import random
import json
import pandas as pd
import numpy as np
import re

DATA_DIR = "data"


def load_coverage_data(coverage_file):
    """Load and parse coverage data from a JSON file."""
    with open(coverage_file, 'r') as f:
        return json.load(f)


def is_function_or_class_declaration(line):
    """Check if a line starts a function or class definition."""
    return bool(re.match(r'^\s*(def |class )', line))


def is_decorator(line):
    """Check if a line is a decorator."""
    return line.strip().startswith("@")


def is_indented(line):
    """Check if a line is indented, indicating it may belong to a previous block."""
    return line.startswith("    ")  # Adjust according to your code indentation style


def get_indentation_level(line):
    """Return the indentation level of a line."""
    return len(line) - len(line.lstrip())


def contains_function_definition_with_body(lines):
    """Check if there is at least one function definition followed by an indented body."""
    for i, line in enumerate(lines):
        if is_function_or_class_declaration(line):
            # Check if the next line is indented, indicating a function body
            if i + 1 < len(lines) and is_indented(lines[i + 1]):
                return True
    return False


def randomly_select_middle_section(file_lines, executed_lines, min_middle=1, max_middle=10):
    """Randomly select a middle section from executed lines, biased towards the center,
    ensuring it does not start with a standalone function or class definition,
    does not include lines with decorators, and does not match the indentation level of defined functions in prefix and suffix."""
    if len(executed_lines) < min_middle:
        return None  # Not enough lines for a middle section

    # Identify indentation levels of function definitions in prefix
    prefix_function_indents = [get_indentation_level(line) for line in file_lines[:executed_lines[0]]
                               if is_function_or_class_declaration(line)]

    for _ in range(10):  # Attempt multiple times if necessary
        # Calculate center position for bias towards middle
        mean_index = len(executed_lines) // 2
        std_dev = len(executed_lines) // 4  # Adjust this to control spread

        # Generate a random starting index using a normal distribution centered at `mean_index`
        start_index = int(np.clip(np.random.normal(mean_index, std_dev), 0, len(executed_lines) - min_middle))
        end_index = min(start_index + random.randint(min_middle, max_middle), len(executed_lines) - 1)

        middle_lines = list(range(executed_lines[start_index], executed_lines[end_index]))

        # Ensure the middle section does not contain any standalone `def` or `class` declarations,
        # does not include decorators, and does not match the indentation of functions in prefix or suffix
        if middle_lines:
            invalid_section = False
            middle_indent = get_indentation_level(file_lines[middle_lines[0] - 1])
            for line_number in middle_lines:
                line = file_lines[line_number - 1]  # Get the actual line content
                if is_function_or_class_declaration(line) or \
                        (get_indentation_level(line) in prefix_function_indents) or \
                        is_decorator(line) or \
                        middle_indent in prefix_function_indents:
                    invalid_section = True
                    break

            if not invalid_section:
                return middle_lines

    # Return None if we couldn't select a valid middle section after multiple attempts
    return None


def split_code(file_lines, middle_lines):
    """Split the code into prefix, middle, and suffix based on selected executed and middle lines."""
    prefix = [line for i, line in enumerate(file_lines, 1) if i < min(middle_lines)]
    middle = [line for i, line in enumerate(file_lines, 1) if i in middle_lines]
    suffix = [line for i, line in enumerate(file_lines, 1) if i > max(middle_lines)]

    # Check if the prefix contains at least one function definition with a body
    if not contains_function_definition_with_body(prefix):
        return None, None, None  # Return None if the prefix doesn't have a function with a body

    return "".join(prefix), "".join(middle), "".join(suffix)


def generate_random_examples(coverage_data, num_examples=50, max_retries=100):
    """Generates random examples from the coverage data with retries for valid splits."""
    examples = []
    all_files = list(coverage_data['files'].items())  # Get all files from the coverage data

    retries = 0
    while len(examples) < num_examples and retries < max_retries:
        # Randomly select a file from the coverage data
        file_path, file_data = random.choice(all_files)
        executed_lines = file_data.get('executed_lines', [])

        if not executed_lines:
            retries += 1
            continue  # Skip if no executed lines are present

        with open(file_path, 'r') as f:
            file_lines = f.readlines()

        # Randomly select a middle section with a central bias
        middle_lines = randomly_select_middle_section(file_lines, executed_lines)
        if not middle_lines:
            retries += 1
            continue  # Retry if unable to select a valid middle section

        # Split the file into prefix, middle, and suffix
        prefix, middle, suffix = split_code(file_lines, middle_lines)

        # Add example if there's a meaningful middle section
        if prefix and middle.strip():
            examples.append({
                "file_path": file_path,
                "prefix": prefix,
                "middle": middle,
                "suffix": suffix
            })
            retries = 0  # Reset retries if a valid example is added
        else:
            retries += 1

    return examples


def save_examples_to_csv(examples, output_file):
    """Save the examples to a CSV file."""
    df = pd.DataFrame(examples)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(examples)} examples to {output_file}")


if __name__ == "__main__":
    coverage_file = os.path.join(DATA_DIR, "combined_coverage.json")  # JSON file with coverage data
    output_file = os.path.join(DATA_DIR, "code_completion_examples.csv")  # Output CSV file
    num_examples = 50  # Target number of examples

    # Load coverage data
    coverage_data = load_coverage_data(coverage_file)

    # Generate random examples by reusing files and sections with retry logic
    examples = generate_random_examples(coverage_data, num_examples)

    # Save the results to a CSV file in the data directory
    save_examples_to_csv(examples, output_file)
