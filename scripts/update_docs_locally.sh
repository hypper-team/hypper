#!/bin/bash
source ./venv/bin/activate
pdoc -o docs/ -d google hypper/ --favicon https://github.com/hypper-team/hypper/raw/main/logo/favicon.png --logo https://github.com/hypper-team/hypper/raw/main/logo/logo1.png --logo-link https://hypper-team.github.io/hypper.html --footer-text "Hypper API Documentation"
cp -r docs/* ../hypper-team.github.io/
cd ../hypper-team.github.io/
git add .
git commit -m "Documentation updates"
git push
