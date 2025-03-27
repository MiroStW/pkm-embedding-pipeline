#!/usr/bin/env python3
"""
CLI tool for managing Git hooks.

This module provides a command-line interface for installing, checking, and
managing Git hooks for the embedding pipeline.
"""

import argparse
import logging
import os
import sys

from src.git_hooks.installer import install_hooks, check_hooks_status
from src.git_hooks.change_detector import GitChangeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("git_hooks.cli")

def install_command(args):
    """
    Install Git hooks.

    Args:
        args: Command-line arguments.
    """
    try:
        installed_hooks = install_hooks(args.repo_path)
        logger.info(f"Successfully installed Git hooks: {', '.join(os.path.basename(h) for h in installed_hooks)}")
    except Exception as e:
        logger.error(f"Failed to install Git hooks: {e}")
        sys.exit(1)

def status_command(args):
    """
    Check the status of Git hooks.

    Args:
        args: Command-line arguments.
    """
    try:
        hook_status = check_hooks_status(args.repo_path)

        # Print the status
        logger.info("Git hook status:")
        for hook_name, installed in hook_status.items():
            status = "Installed" if installed else "Not installed"
            logger.info(f"  {hook_name}: {status}")

        # Check if all hooks are installed
        if all(hook_status.values()):
            logger.info("All hooks are installed.")
        else:
            missing_hooks = [hook for hook, installed in hook_status.items() if not installed]
            logger.warning(f"Some hooks are missing: {', '.join(missing_hooks)}")
            logger.info("Run 'python -m src.git_hooks.cli install' to install missing hooks.")
    except Exception as e:
        logger.error(f"Failed to check hook status: {e}")
        sys.exit(1)

def list_files_command(args):
    """
    List all markdown files tracked by Git.

    Args:
        args: Command-line arguments.
    """
    try:
        detector = GitChangeDetector(args.repo_path)
        files = detector.get_all_markdown_files()

        logger.info(f"Found {len(files)} markdown files tracked by Git:")
        for file in files:
            logger.info(f"  - {file}")
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        sys.exit(1)

def main():
    """Main entry point for the Git hooks CLI."""
    parser = argparse.ArgumentParser(description="Git hooks management for the embedding pipeline")
    parser.add_argument("--repo-path", help="Path to the Git repository. If not specified, the current directory is used.")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Install command
    install_parser = subparsers.add_parser("install", help="Install Git hooks")
    install_parser.set_defaults(func=install_command)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check the status of Git hooks")
    status_parser.set_defaults(func=status_command)

    # List files command
    list_files_parser = subparsers.add_parser("list-files", help="List all markdown files tracked by Git")
    list_files_parser.set_defaults(func=list_files_command)

    # Parse arguments
    args = parser.parse_args()

    # Run the appropriate command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()