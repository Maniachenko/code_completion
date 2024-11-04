# code_completion

This project aims to create a dataset of code completion examples, generate completions using a model, and evaluate the results against the original code snippets. The goal is to analyze the model's performance and propose suitable evaluation metrics.

## Project Overview

1. **Dataset Creation**: Generate a dataset of code snippets from a personal project, simulating the "user cursor position" with three parts—`prefix`, `middle`, and `suffix`.
2. **Model Inference**: Use an open-source code completion model (e.g., Tiny Starcoder) to predict completions for the dataset examples.
3. **Evaluation**: Manually review and automatically evaluate the generated completions using metrics like exact match, ChrF, and others.

## Directory Structure

```plaintext
data/                    # Directory for storing intermediate and final data files
README.md                # Project description and instructions
code_generation.py       # Script for generating code completions with model
extract_data_from_repo.py# Script for extracting code completion examples based on coverage data
metrics_comparison.ipynb # Notebook for evaluating and comparing metrics
requirements.txt         # List of dependencies for the project
run_tests_coverage.py    # Script for running tests and generating coverage reports
```

## 1. Dataset Creation

The `run_tests_coverage.py` script is designed to analyze and collect code coverage data from multiple repositories. This data will later be used to create a code completion dataset by identifying actively used and tested portions of the codebase. Below is a breakdown of the script’s core principles and workflow.

## Workflow

1. **Dynamic File Inclusion and Exclusion**:
   - The script reads `.gitignore` files from each repository to identify and exclude specific files or directories (e.g., `__pycache__` and `.pytest_cache`).
   - Additional ignore patterns are included to avoid non-code files, reducing unnecessary analysis and storage use.

2. **Modular Coverage Data Collection**:
   - The script checks each repository for a Python virtual environment, activating it if present. This ensures dependencies like `coverage` and `pytest` are installed within the environment.
   - By running tests within each environment, the script generates a JSON-format coverage report, capturing line-by-line code execution details.

3. **Automated Code Completion Checks Using Unit Tests**:
   - Unit tests present in the project repositories are leveraged to automate the evaluation of generated code completions.
   - By comparing the results of unit tests on the generated code with those of the original code, the script can provide automated feedback on the correctness and completeness of code completion predictions.

4. **Unified Data Aggregation**:
   - Individual coverage reports from each repository are combined into a single, comprehensive JSON file.
   - This combined report provides a unified view of code usage across repositories, essential for selecting meaningful code examples for the dataset.

The `extract_data_from_repo.py` script is designed to create a dataset of code completion examples based on coverage data collected from multiple repositories. This data provides line-by-line information on executed code, which the script uses to identify meaningful sections for generating realistic code completion examples. The script splits selected code sections into three parts—prefix, middle, and suffix—to simulate the "user cursor position" and allow for completion predictions by a code model. Below is a breakdown of the script’s core principles and workflow.

## Workflow

1. **Loading Coverage Data**:
   - The script begins by loading coverage data from combined_coverage.json. This data provides information on executed lines (covered by tests) in each file, which helps identify meaningful code sections to use for creating examples.

2. **Identification of Code Structure and Validity Checks**:
   - Various helper functions parse the code structure by identifying functions, classes, and decorators.
   - Additional checks ensure code sections meet specific requirements (e.g., avoiding function or class headers, checking indentation levels), helping select valid and coherent code sections.

3. **Random Selection of Middle Code Section**:
   - For each file, the script randomly selects a middle section from executed lines, simulating the position of a user cursor.
   - This middle section is chosen near the center and is validated to exclude any function or class declarations or decorators. This ensures a realistic missing code section for model inference.

4. **Splitting Code into Prefix, Middle, and Suffix**:
   - The code is split into three parts: prefix (code before the cursor), middle (code to be completed), and suffix (code after the cursor).
   - Only examples with a valid prefix (containing at least one function with a body) are included in the dataset. This structure imitates a real coding situation where a developer has partially written code and is in the process of completing a function or block, providing relevant context for code completion.

5. **Saving Examples to CSV**:
   - The valid examples, including file paths and the prefix, middle, and suffix sections, are saved to a CSV file (code_completion_examples.csv).
   - This structured dataset provides a realistic basis for model inference by capturing missing code segments in context.

## 2. Model Inference

The code_generation.py script is responsible for generating code completions by leveraging a pretrained code completion model, such as Tiny Starcoder, configured with quantization for efficiency. This script reads from a structured dataset of code completion examples, applies Fill-In-the-Middle (FIM) formatting, and generates completions for missing code segments based on the provided context. Below is a breakdown of the script's core principles and workflow.
Workflow

1. **Loading and Configuring the Model**:
   - The script loads the Tiny Starcoder model and tokenizer with an 8-bit quantization configuration to optimize memory usage.
   - The model is set to run on a GPU if available, ensuring efficient computation.
   - Key generation parameters include `max_total_length = 4096` to limit the length of the prompt after FIM formatting and `max_new_tokens = 128` to restrict the length of the generated completion.

2. **Formatting Input for FIM**:
   - FIM formatting is applied by combining the prefix and suffix from each example in the dataset.
   - The format_fim_input function structures the prompt with <fim_prefix>, <fim_suffix>, and <fim_middle> (`<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>`) tokens to clearly mark the missing segment’s position.

3. **Truncating Context**:
   - The truncate_middle_context function ensures the prompt fits within the model’s token limit by trimming excess context around the suffix marker.
   - This balanced truncation preserves as much surrounding context as possible to aid the model’s generation.

4. **Removing Repetitive Subsequence**:
   - The remove_subsequences_from_output function identifies any subsequences from the prompt in the generated output and removes them to avoid redundancy.
   - This step is essential because Tiny Starcoder, due to the structure of the FIM prompt, often generates the middle part at the end of the output. Removing repetitive subsequences ensures that the output is novel and focuses solely on completing the missing segment rather than repeating portions of the prompt.

5. **Generating Code Completions**:
   - The generate_code function iterates through each dataset entry, generating a completion for the missing code segment and storing it in the specified column.
        Generation settings are configured to allow for creative, varied outputs, using temperature, top-k, and top-p sampling techniques.
        The generated code is printed for each example to allow tracking of progress and debugging.

6. **Saving Results**:
   - The dataset, now with generated completions, is saved to a CSV file for further analysis and evaluation.

## 3. Evaluation

The metrics_comparison.ipynb notebook is used for evaluating the quality of generated code completions. This file combines both manual scoring and automatic evaluation metrics to analyze how well the generated code matches the reference code. Here’s a detailed breakdown of its workflow and contents:

1. **Imports and Setup**:
   - Necessary libraries such as NLTK, SacreBLEU, and Rouge Score are imported.
   - Required resources for tokenization and word processing are downloaded for NLTK.

2. **Loading and Displaying Data**:
   - The notebook reads the code completions dataset (data/code_completion_output.csv), displaying columns like file_path, prefix, middle, suffix, and generated_code.
   - Manual Scoring:
        A loop iterates through each example in the dataset, displaying the original middle section and the generated completion.
        The evaluator assigns a manual score to each instance, with 0 for no similarity, 0.5 for partial similarity, and 1 for high similarity.
        The manual score is then appended to the dataset.

3. **Automated Metric Calculation**:
   - The notebook calculates several metrics to compare the generated code against the reference middle code section. The metrics include:
            * Exact Match: Binary metric that checks for identical matches.
            * BLEU: Measures n-gram overlap, with smoothing to handle cases with fewer matches.
            * ROUGE-L: Focuses on the longest common subsequence to capture structural similarity.
            * METEOR: Evaluates both exact matches and semantic similarity through stemming.
            * ChrF: Character-level metric that compares entire code blocks, useful for evaluating structural variations.

4. **Correlation Analysis**:
   - The notebook calculates Spearman and Kendall correlation coefficients between the manual scores and each automated metric to identify which metrics best align with human judgment.
   - Results indicate that BLEU and ROUGE-L have the highest correlations, suggesting that these metrics best capture the similarity aspects valued in the manual evaluations.
  
## Conclusion:

The code completion results were less accurate than anticipated, with no instances of exact code matches across the generated completions. A closer review revealed that many of the generated middle segments lacked sufficient contextual alignment with the reference, making them unsuitable for more advanced functional evaluations, such as unit tests, which could serve as a definitive quality check in future iterations.

This outcome suggests several areas for potential improvement:

- **Prompt Structure Refinemen**t: The Fill-In-the-Middle (FIM) prompt format used in this project could be further optimized. By experimenting with prompt variations, such as adjusting the balance between prefix and suffix context or employing different markers, we may improve the model’s ability to generate accurate, contextually relevant completions.

- **Model Scaling**: Exploring larger, more advanced models could increase the likelihood of producing coherent completions, particularly for complex or nuanced code segments. Models with more parameters and extensive training on code-specific data may better capture intricate code dependencies and conventions.

| Metric   | Spearman Correlation | Kendall Correlation |
|----------|-----------------------|---------------------|
| BLEU     | 0.613                | 0.498              |
| ROUGE-L  | 0.612                | 0.495              |
| METEOR   | 0.488                | 0.380              |
| ChrF     | -0.348               | -0.322             |

The correlation analysis indicates that BLEU and ROUGE-L metrics exhibit the strongest relationship with human judgments, suggesting that they best capture aspects of code similarity relevant to manual evaluation. Specifically, BLEU and ROUGE-L emphasize structural and lexical overlap, making them suitable for tasks where exact phrase matching and structural continuity are valued. This insight points toward BLEU and ROUGE-L as primary metrics for evaluating code completion, particularly when measuring syntactic fidelity and structural coherence.

In future work, I can further explore these metrics in tandem with additional evaluation techniques, including unit test validation and more contextually aware metrics, to provide a more comprehensive assessment of model quality in code completion tasks.
