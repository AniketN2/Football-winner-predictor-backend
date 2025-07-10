#!/bin/bash
# Exit immediately if any command fails
set -e

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip and install wheel first
pip install --upgrade pip wheel

# Install requirements with no cache and force-reinstall
pip install --no-cache-dir --force-reinstall -r requirements.txt

# Verify installations
pip freeze
