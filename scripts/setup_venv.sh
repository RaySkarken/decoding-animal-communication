#!/usr/bin/env bash
# Create .venv and install deps. Run from project root:  bash scripts/setup_venv.sh
# Never installs outside the venv. Uses staged install (core → tensorflow → GitHub) for reliability.

set -e
cd "$(dirname "$0")/.."

if [[ -d .venv ]]; then
  echo ".venv already exists. To (re)install deps run: bash scripts/finish_install.sh"
  echo "Activate with: source .venv/bin/activate"
  exit 0
fi

echo "Creating .venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies into .venv (staged)..."
pip install --upgrade pip
pip install -r requirements-core.txt
pip install "tensorflow>=2.13"
pip install tensorflow-metal 2>/dev/null || true
pip install git+https://github.com/timsainb/AVGN.git git+https://github.com/timsainb/vocalization-segmentation.git

echo "Done. Activate with: source .venv/bin/activate"
