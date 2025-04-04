"""
This file contains pytest configuration and fixtures.
It's automatically loaded by pytest when running tests.
"""
import os
import sys

# Add the parent directory to sys.path to allow importing geometrize module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))