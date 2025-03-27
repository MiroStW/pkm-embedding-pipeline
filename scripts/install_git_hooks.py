#!/usr/bin/env python
"""
Script to install git hooks for automatic tracking of document changes.
"""
import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Install git hooks for document tracking')

    parser.add_argument('--git-dir',
                        help='Path to the git repository (defaults to current directory)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing hooks')
    parser.add_argument('--uninstall', action='store_true',
                        help='Uninstall hooks instead of installing them')

    return parser.parse_args()

def get_hooks_dir(git_dir):
    """Get the path to the git hooks directory."""
    # First check if .git is a directory
    hooks_dir = os.path.join(git_dir, '.git', 'hooks')
    if os.path.isdir(hooks_dir):
        return hooks_dir

    # If .git is a file (submodule or worktree), read the gitdir from it
    git_file = os.path.join(git_dir, '.git')
    if os.path.isfile(git_file):
        with open(git_file, 'r') as f:
            gitdir_line = f.readline().strip()
            if gitdir_line.startswith('gitdir:'):
                gitdir = gitdir_line[7:].strip()
                hooks_dir = os.path.join(git_dir, gitdir, 'hooks')
                if os.path.isdir(hooks_dir):
                    return hooks_dir

    return None

def backup_hook(hook_path):
    """Backup an existing hook."""
    if os.path.exists(hook_path):
        backup_path = f"{hook_path}.bak"
        logging.info(f"Backing up existing hook to {backup_path}")
        shutil.copy2(hook_path, backup_path)

def install_hooks(hooks_dir, force=False):
    """Install the git hooks to the specified directory."""
    # Get source hooks
    src_hooks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'git_hooks')

    hooks_to_install = {
        'post-commit': 'post-commit.py',
        'post-merge': 'post-merge.py',
        'post-checkout': 'post-checkout.py'
    }

    installed_hooks = []

    for hook_name, src_file in hooks_to_install.items():
        src_path = os.path.join(src_hooks_dir, src_file)
        dst_path = os.path.join(hooks_dir, hook_name)

        # Check if hook already exists
        if os.path.exists(dst_path) and not force:
            logging.warning(f"Hook {hook_name} already exists. Use --force to overwrite.")
            continue

        # Backup existing hook
        if os.path.exists(dst_path):
            backup_hook(dst_path)

        # Copy the hook
        try:
            shutil.copy2(src_path, dst_path)
            os.chmod(dst_path, 0o755)  # Make executable
            logging.info(f"Installed {hook_name} hook")
            installed_hooks.append(hook_name)
        except Exception as e:
            logging.error(f"Failed to install {hook_name} hook: {str(e)}")

    return installed_hooks

def uninstall_hooks(hooks_dir):
    """Uninstall the git hooks."""
    hooks_to_uninstall = ['post-commit', 'post-merge', 'post-checkout']
    uninstalled_hooks = []

    for hook_name in hooks_to_uninstall:
        hook_path = os.path.join(hooks_dir, hook_name)
        backup_path = f"{hook_path}.bak"

        if os.path.exists(hook_path):
            # Check if it's our hook (contains pkm-chatbot)
            with open(hook_path, 'r') as f:
                content = f.read()
                if 'pkm-chatbot' in content:
                    try:
                        # Remove the hook
                        os.remove(hook_path)
                        logging.info(f"Removed {hook_name} hook")
                        uninstalled_hooks.append(hook_name)

                        # Restore backup if it exists
                        if os.path.exists(backup_path):
                            shutil.copy2(backup_path, hook_path)
                            os.chmod(hook_path, 0o755)
                            os.remove(backup_path)
                            logging.info(f"Restored original {hook_name} hook from backup")
                    except Exception as e:
                        logging.error(f"Failed to uninstall {hook_name} hook: {str(e)}")
                else:
                    logging.warning(f"Hook {hook_name} does not appear to be a PKM Chatbot hook. Skipping.")

    return uninstalled_hooks

def main():
    """Main entry point for the script."""
    setup_logging()
    args = parse_args()

    # Determine git repository directory
    git_dir = args.git_dir or os.getcwd()

    logging.info(f"Working with git repository at: {git_dir}")

    # Find hooks directory
    hooks_dir = get_hooks_dir(git_dir)
    if not hooks_dir:
        logging.error(f"Could not find git hooks directory for {git_dir}")
        return 1

    logging.info(f"Found git hooks directory: {hooks_dir}")

    if args.uninstall:
        # Uninstall hooks
        uninstalled = uninstall_hooks(hooks_dir)
        if uninstalled:
            logging.info(f"Successfully uninstalled {len(uninstalled)} hooks: {', '.join(uninstalled)}")
            return 0
        else:
            logging.warning("No hooks were uninstalled")
            return 1
    else:
        # Install hooks
        installed = install_hooks(hooks_dir, args.force)
        if installed:
            logging.info(f"Successfully installed {len(installed)} hooks: {', '.join(installed)}")

            # Print instructions
            logging.info("\nTo use these hooks:")
            logging.info("1. Make sure the Python environment is activated when you're using git")
            logging.info("2. Changes to markdown files will be automatically tracked")
            logging.info("3. Run the pipeline using: python -m src.cli.main process")

            return 0
        else:
            logging.warning("No hooks were installed")
            return 1

if __name__ == "__main__":
    sys.exit(main())