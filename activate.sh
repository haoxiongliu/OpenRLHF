#!/bin/bash
# Activate uv virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Activated uv virtual environment: $(which python)"
else
    echo "Error: .venv/bin/activate not found. Run 'uv venv' to create the environment."
fi
