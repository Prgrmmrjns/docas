"""Sphinx configuration for the DOCAS package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "DOCAS"
author = "Jonas C. Wolber"
copyright = "2026, Jonas C. Wolber"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
autoclass_content = "both"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True

intersphinx_mapping = {"python": ("https://docs.python.org/3", None), "numpy": ("https://numpy.org/doc/stable", None)}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "DOCAS"
