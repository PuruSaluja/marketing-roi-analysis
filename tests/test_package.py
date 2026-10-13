import tomllib
from pathlib import Path

import marketing_roi


def test_the_version_is_the_one_pyproject_declares() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert marketing_roi.__version__ == declared
