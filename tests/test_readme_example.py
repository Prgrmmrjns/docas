"""Execute the README quickstart block (must stay in sync with README.md)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def _readme_python_block() -> str:
    text = README.read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks, "no python fence in README"
    # First fence is the quickstart.
    return blocks[0]


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_readme_quickstart_runs(capsys):
    code = _readme_python_block()
    ns: dict = {}
    exec(compile(code, str(README), "exec"), ns, ns)
    out = capsys.readouterr().out
    assert "align. error" in out
    assert "aligned" in out
