import pytest
import pkg_resources
from pathlib import Path


@pytest.fixture()
def requirements_path():
    return Path(__file__).parents[1] / "requirements.txt"


def test_requirements(requirements_path):
    # Test required packages availibility
    with open(requirements_path, "r") as f:
        reqs_from_file = pkg_resources.parse_requirements(f.read())
    for req in reqs_from_file:
        pkg_resources.require(str(req))
