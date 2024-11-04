import os
import subprocess
import fnmatch
import json

DATA_DIR = "data"


def parse_gitignore(repo_path):
    """
    Parse the .gitignore file to get a list of patterns for ignored files and directories.
    Adds common ignore patterns like __pycache__ and tests to the list.
    """
    gitignore_path = os.path.join(repo_path, '.gitignore')
    ignore_patterns = []
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines, add patterns to ignore list
                if line and not line.startswith('#'):
                    pattern_path = os.path.join(repo_path, line) if not os.path.isabs(line) else line
                    ignore_patterns.append(pattern_path)
    # Add common patterns to ignore list
    ignore_patterns.extend(['*__pycache__*', '*.pytest_cache*', '.*', '__*__', 'tests'])
    return ignore_patterns


def should_ignore(path, ignore_patterns):
    """
    Determine if a given path should be ignored based on ignore patterns.
    Checks both the absolute path and the basename for matches.
    """
    abs_path = os.path.abspath(path)
    for pattern in ignore_patterns:
        # Check if the path or its basename matches any ignore pattern
        if fnmatch.fnmatch(abs_path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def run_tests_with_coverage(repo_path, omit_patterns=None):
    """
    Run tests with coverage in the specified repository path, creating a coverage JSON report.
    Uses virtual environment if it exists in the repo path, otherwise skips testing in that path.
    """
    ignore_patterns = parse_gitignore(repo_path)
    if should_ignore(repo_path, ignore_patterns):
        print(f"Skipping ignored path: {repo_path}")
        return

    # Determine virtual environment name
    venv_name = 'airflow_venv' if 'airflow' in repo_path else 'venv'
    venv_path = os.path.join(repo_path, venv_name)
    if not os.path.isdir(venv_path):
        print(f"No virtual environment found at {venv_path}. Skipping {repo_path}.")
        return

    # Change to the repository directory
    original_dir = os.getcwd()
    os.chdir(repo_path)

    # Prepare command to activate virtual environment and install dependencies if needed
    activate_script = os.path.join(venv_name, 'bin', 'activate')
    command = f"source {activate_script} && "
    command += "pip show coverage || pip install coverage && "
    command += "pip show pytest || pip install pytest && "

    # Format omit patterns for coverage command
    omit_arg = ','.join(omit_patterns) if omit_patterns else ""

    # Run tests with coverage and generate a JSON report saved to the DATA_DIR
    coverage_json_path = os.path.join(original_dir, DATA_DIR, f"{os.path.basename(repo_path)}_coverage.json")
    print(f"Running tests with coverage in {repo_path}")
    command += f'coverage run -m pytest && coverage json --omit={omit_arg} --pretty-print -o {coverage_json_path}'

    subprocess.run(command, shell=True, executable="/bin/bash")
    os.chdir(original_dir)


def merge_json_reports(repo_paths):
    """
    Combine individual coverage JSON reports from each repo path into a single report.
    Saves the combined report to the DATA_DIR.
    """
    combined_data = {"meta": {}, "files": {}}
    for repo_path in repo_paths:
        # Look for each repo's coverage JSON report in the DATA_DIR
        coverage_file = os.path.join(DATA_DIR, f"{os.path.basename(repo_path)}_coverage.json")
        if os.path.exists(coverage_file):
            with open(coverage_file, "r") as f:
                data = json.load(f)

                # Update meta information with data from current report
                combined_data["meta"].update(data.get("meta", {}))

                # Add file data with repo path prefix for uniqueness
                for file_path, file_data in data.get("files", {}).items():
                    combined_path = os.path.join(repo_path, file_path)
                    combined_data["files"][combined_path] = file_data
        else:
            print(f"No coverage.json found for {repo_path}")

    # Save combined coverage report to the DATA_DIR
    combined_coverage_path = os.path.join(DATA_DIR, "combined_coverage.json")
    with open(combined_coverage_path, "w") as f:
        json.dump(combined_data, f, indent=4)
    print(f"Combined coverage report saved to {combined_coverage_path}")


if __name__ == "__main__":
    # Define paths to each repository
    repo_paths = [
        "../airflow",
        "../PycharmProjects/sales_telegram_bot/backend/sales_telegram_bot_admin_backend",
        "../PycharmProjects/sales_telegram_bot/backend/telegram_lambda_package",
        "../PycharmProjects/sales_telegram_bot/backend/models_app"
    ]

    # Define patterns to omit from coverage analysis
    omit_patterns = [
        "*/config-3.py",
        "*/config.py",
        "*/_remote_module_non_scriptable.py",
        "*/_remote_module_non_scriptable.pyc",
        "*/.cache/*",
        "*/.huggingface/*",
        "*/tests/*",
        "*/test_*.py"
    ]

    # Run tests with coverage in each repository path and merge results
    for repo_path in repo_paths:
        run_tests_with_coverage(repo_path, omit_patterns)
    merge_json_reports(repo_paths)
