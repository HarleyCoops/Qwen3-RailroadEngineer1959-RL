"""
Dakota1890 package - Re-export from dakota_grammar_translation for compatibility.
This allows both import styles:
- from dakota_grammar_translation import load_environment
- from dakota1890 import load_environment
"""

from dakota_grammar_translation import load_environment

__all__ = ["load_environment"]

