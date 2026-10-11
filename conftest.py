"""
Makes the src/ package importable from tests/ regardless of how pytest
is invoked. Run pytest from the project root:

    pytest tests/ -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
