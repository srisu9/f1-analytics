"""
conftest.py
============
Pytest configuration for the F1 Analytics test suite.

Makes the `src` and `app` directories importable during testing
without requiring the package to be installed.
"""
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
