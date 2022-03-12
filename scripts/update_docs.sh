#!/bin/bash
source ./venv/bin/activate
pdoc -o docs/ -d google hypper/
cp -r docs/* ../hypper-team.github.io/
cd ../hypper-team.github.io/
git add .
git commit -m "Documentation updates"
git push
