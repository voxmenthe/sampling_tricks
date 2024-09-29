#!/bin/bash

# Install the project in editable mode
pip install -e .

# Create and install the IPython kernel for the project
python -m ipykernel install --user --name=sampling_tricks --display-name "Sampling Tricks"

echo "Jupyter kernel 'Sampling Tricks' has been installed."