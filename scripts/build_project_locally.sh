#!/bin/bash
python3 -m pip install build twine
python3 -m build
python3 -m twine upload --repository $1 dist/hypper-$2*