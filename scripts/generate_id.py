#!/usr/bin/env python
"""
Simple script to generate a unique ID.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cli.utils import generate_id

if __name__ == "__main__":
    new_id = generate_id()
    print(f"Generated ID: {new_id}")