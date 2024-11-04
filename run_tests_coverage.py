import os
import subprocess
import fnmatch
import json


def parse_gitignore(repo_path):
    gitignore_path = os.path.join(repo_path, '.gitignore')
    ignore_patterns = []
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    pattern_path = os.path.join(repo_path, line) if not os.path.isabs(line) else line
                    ignore_patterns.append(pattern_path)
    ignore_patterns.extend(['*__pycache__*', '*.pytest_cache*', '.*', '__*__', 'tests'])
    return ignore_patterns


def should_ignore(path, ignore_patterns):
    abs_path = os.path.abspath(path)
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(abs_path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def run_tests_with_coverage(repo_path, omit_patterns=None):
    ignore_patterns = parse_gitignore(repo_path)
    if should_ignore(repo_path, ignore_patterns):
        print(f"Skipping ignored path: {repo_path}")
        return

    venv_name = 'airflow_venv' if 'airflow' in repo_path else 'venv'
    venv_path = os.path.join(repo_path, venv_name)
    if not os.path.isdir(venv_path):
        print(f"No virtual environment found at {venv_path}. Skipping {repo_path}.")
        return

    original_dir = os.getcwd()
    os.chdir(repo_path)

    activate_script = os.path.join(venv_name, 'bin', 'activate')
    command = f"source {activate_script} && "
    command += "pip show coverage || pip install coverage && "
    command += "pip show pytest || pip install pytest && "

    # Format omit patterns properly for the command
    omit_arg = ','.join(omit_patterns) if omit_patterns else ""

    print(f"Running tests with coverage in {repo_path}")
    command += f'coverage run -m pytest && coverage json --omit={omit_arg} --pretty-print -o coverage.json'

    subprocess.run(command, shell=True, executable="/bin/bash")
    os.chdir(original_dir)


def merge_json_reports(repo_paths):
    combined_data = {"meta": {}, "files": {}}
    for repo_path in repo_paths:
        coverage_file = os.path.join(repo_path, "coverage.json")
        if os.path.exists(coverage_file):
            with open(coverage_file, "r") as f:
                data = json.load(f)

                # Update meta information
                combined_data["meta"].update(data.get("meta", {}))

                # Add files with repo path prefix
                for file_path, file_data in data.get("files", {}).items():
                    combined_path = os.path.join(repo_path, file_path)
                    combined_data["files"][combined_path] = file_data
        else:
            print(f"No coverage.json found in {repo_path}")

    # Save combined JSON data
    with open("combined_coverage.json", "w") as f:
        json.dump(combined_data, f, indent=4)
    print("Combined coverage report saved to combined_coverage.json")


if __name__ == "__main__":
    repo_paths = [
        "../airflow",
        "../PycharmProjects/sales_telegram_bot/backend/sales_telegram_bot_admin_backend",
        "../PycharmProjects/sales_telegram_bot/backend/telegram_lambda_package",
        "../PycharmProjects/sales_telegram_bot/backend/models_app"
    ]

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

    for repo_path in repo_paths:
        run_tests_with_coverage(repo_path, omit_patterns)
    merge_json_reports(repo_paths)
