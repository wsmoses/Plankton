#!/bin/bash

python -m venv venv

source ./venv/bin/activate

pip install ./dace
pip install ./daceml
pip install -e .


echo "Installation complete. To use, please activate virtualenv by executing 'source ./venv/bin/activate'"

