#!/usr/bin/env bash
# Finish installing into existing .venv (e.g. if first run was interrupted).
# Run from project root:  bash scripts/finish_install.sh

set -e
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "No .venv found. Run: bash scripts/setup_venv.sh"
  exit 1
fi

source .venv/bin/activate
echo "Installing into .venv..."
pip install --upgrade pip
pip install -r requirements-core.txt
pip install "tensorflow>=2.13"
pip install tensorflow-metal 2>/dev/null || true
pip install git+https://github.com/timsainb/AVGN.git git+https://github.com/timsainb/vocalization-segmentation.git
echo "Done. Activate with: source .venv/bin/activate"
