#!/usr/bin/env python3
"""
Git hook trigger script.

This script is called by the git hooks to process changes and
trigger the embedding pipeline.
"""

import argparse
import logging
import os
import sys
from typing import List, Dict

# Add the parent directory to the path so we can import the embedding pipeline modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.git_hooks.change_detector import GitChangeDetector
from src.database.document_db import DocumentTracker  # Import the tracking database

# Ensure logs directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "git_hooks.log"), mode="a")
    ]
)
logger = logging.getLogger("git_hooks.trigger")

def process_changes(hook_type: str, repo_path: str = None) -> Dict[str, List[str]]:
    """
    Process changes detected by the git hook.

    Args:
        hook_type: Type of the hook (post-commit, post-merge).
        repo_path: Path to the Git repository. If None, the current directory is used.

    Returns:
        Dictionary with keys 'added', 'modified', and 'deleted', each containing a list of file paths.
    """
    detector = GitChangeDetector(repo_path)
    tracker = DocumentTracker()  # Initialize the tracking database

    # Detect changes based on the hook type
    if hook_type == "post-commit":
        changes = detector.get_last_commit_changes()
    elif hook_type == "post-merge":
        changes = detector.get_merge_changes()
    else:
        logger.error(f"Unknown hook type: {hook_type}")
        return {'added': [], 'modified': [], 'deleted': []}

    # Log the changes
    logger.info(f"Changes detected by {hook_type} hook:")
    for change_type, files in changes.items():
        if files:
            logger.info(f"  {change_type.capitalize()}: {len(files)} files")
            for file in files:
                logger.info(f"    - {file}")

    # Update the document tracking database
    if changes['added'] or changes['modified']:
        # Mark added and modified files for processing
        for file_path in changes['added'] + changes['modified']:
            tracker.mark_for_processing(file_path)

    if changes['deleted']:
        # Mark deleted files for deletion in the vector database
        for file_path in changes['deleted']:
            tracker.mark_for_deletion(file_path)

    return changes

def main():
    """Main entry point for the git hook trigger script."""
    parser = argparse.ArgumentParser(description="Git hook trigger for the embedding pipeline")
    parser.add_argument("hook_type", choices=["post-commit", "post-merge"], help="Type of the hook")
    parser.add_argument("--repo-path", help="Path to the Git repository")

    args = parser.parse_args()

    try:
        changes = process_changes(args.hook_type, args.repo_path)

        # If changes were detected, we should signal that the pipeline needs to run
        # This could be done by writing to a file, sending a signal, etc.
        if any(changes.values()):
            # This is a placeholder, in a real implementation you might want to
            # trigger the pipeline to run right away or notify a daemon
            logger.info("Changes detected, pipeline needs to run")

            # Create logs directory if it doesn't exist
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
            os.makedirs(log_dir, exist_ok=True)

            # Write a flag file to indicate that the pipeline needs to run
            with open(os.path.join(log_dir, "pipeline_trigger"), "w") as f:
                f.write(f"{args.hook_type} trigger at {os.path.basename(__file__)}")
        else:
            logger.info("No changes detected, no need to run the pipeline")

    except Exception as e:
        logger.exception(f"Error processing changes: {e}")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()