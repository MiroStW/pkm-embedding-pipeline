#!/usr/bin/env python3
"""
Git hook script templates for the embedding pipeline.

These templates are used to create the actual hook scripts that will be installed
in the .git/hooks directory of the repository.
"""

POST_COMMIT_HOOK = """#!/bin/bash
# Post-commit hook for the embedding pipeline
# This hook is called after a commit is made and identifies changed files

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# Execute the embedding pipeline script with commit trigger
cd "$REPO_ROOT"
python -m src.git_hooks.trigger post-commit

# Exit with the status of the last command
exit $?
"""

POST_MERGE_HOOK = """#!/bin/bash
# Post-merge hook for the embedding pipeline
# This hook is called after a merge or pull is made

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# Execute the embedding pipeline script with merge trigger
cd "$REPO_ROOT"
python -m src.git_hooks.trigger post-merge

# Exit with the status of the last command
exit $?
"""

# Add more hooks as needed (e.g., post-checkout, post-rewrite)