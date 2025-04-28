"""
Keep context.py minimal, and focused only on setting up the import context.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
