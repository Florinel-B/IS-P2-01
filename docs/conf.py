import os
import sys
from datetime import datetime

# Put the project root (where `src/` lives) on sys.path so autodoc can import it
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

project = 'IS-P2-01'
author = 'Florinel-B'
copyright = f"{datetime.now().year}, {author}"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Enable autosummary generation
autosummary_generate = True

# When building on Read the Docs we often don't want to install heavy deps
# such as torch; mock them so autodoc can import modules that reference them.
autodoc_mock_imports = ['torch']

# Prefer the Read the Docs theme when building on RTD or locally if installed
try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except Exception:
    # fallback
    html_theme = 'alabaster'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# General HTML options
html_theme = 'alabaster'
html_static_path = ['_static']

# Autodoc options
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
