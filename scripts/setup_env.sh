#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep Ray's interpreter path free of spaces; never modify an existing env.
env_path="${1:-$HOME/.venvs/damldg}"
python_bin="${PYTHON:-python3.11}"
if [[ -e "$env_path" ]]; then
  echo "Environment already exists: $env_path; activate it or choose a new path." >&2
  exit 1
fi
if [[ "$env_path" == *" "* ]]; then
  echo "Choose an environment path without spaces." >&2
  exit 1
fi
"$python_bin" -m venv "$env_path"
"$env_path/bin/python" -m pip install --upgrade pip
"$env_path/bin/python" -m pip install -e '.[test]' -c requirements-lock-py311.txt
"$env_path/bin/python" -m pip check
printf 'Activate with: source "%s/bin/activate"\n' "$env_path"
