import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import marketing_roi

ROOT = Path(__file__).resolve().parents[1]


def test_the_installed_version_is_the_package_s() -> None:
    assert version("marketing-roi-analysis") == marketing_roi.__version__


def test_pyproject_takes_its_version_from_the_package() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in pyproject["project"]["dynamic"]
    attr = pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "marketing_roi.__version__"


def test_it_imports_from_a_source_tree_that_was_never_installed(tmp_path: Path) -> None:
    # A copy of the package alone, so no install metadata sits beside it, and
    # -S keeps site-packages, where the installed copy's metadata is, off the path.
    shutil.copytree(ROOT / "src" / "marketing_roi", tmp_path / "marketing_roi")
    code = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); "
        "import marketing_roi; print(marketing_roi.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == marketing_roi.__version__
