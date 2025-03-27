#!/usr/bin/env python3
"""
Git hook installer module.

This module provides functionality to install Git hooks for the embedding pipeline.
"""

import os
import stat
import subprocess
from pathlib import Path
from typing import List, Optional

from src.git_hooks.hooks import POST_COMMIT_HOOK, POST_MERGE_HOOK


class GitHookInstaller:
    """Installs Git hooks for the embedding pipeline."""

    def __init__(self, repo_path: str = None):
        """
        Initialize the GitHookInstaller.

        Args:
            repo_path: Path to the Git repository. If None, the current directory is used.
        """
        self.repo_path = repo_path or os.getcwd()
        self.hooks_dir = self._get_hooks_dir()

    def _get_hooks_dir(self) -> Path:
        """
        Get the path to the Git hooks directory.

        Returns:
            Path to the Git hooks directory.
        """
        try:
            git_dir = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            ).stdout.strip()

            return Path(self.repo_path) / git_dir / "hooks"
        except subprocess.CalledProcessError:
            raise ValueError(f"Not a Git repository: {self.repo_path}")

    def _write_hook(self, hook_name: str, content: str) -> str:
        """
        Write a hook file and make it executable.

        Args:
            hook_name: Name of the hook (e.g., "post-commit").
            content: Content of the hook script.

        Returns:
            Path to the created hook file.
        """
        hook_path = self.hooks_dir / hook_name

        # Create the hook file
        with open(hook_path, "w") as f:
            f.write(content)

        # Make the hook file executable
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        return str(hook_path)

    def install_all_hooks(self) -> List[str]:
        """
        Install all hooks for the embedding pipeline.

        Returns:
            List of paths to the installed hook files.
        """
        installed_hooks = []

        # Install post-commit hook
        installed_hooks.append(self._write_hook("post-commit", POST_COMMIT_HOOK))

        # Install post-merge hook
        installed_hooks.append(self._write_hook("post-merge", POST_MERGE_HOOK))

        return installed_hooks

    def check_hook_status(self) -> dict:
        """
        Check the status of the hooks.

        Returns:
            Dictionary with hook names as keys and boolean indicating if they are installed.
        """
        hooks = {
            "post-commit": False,
            "post-merge": False,
        }

        for hook_name in hooks:
            hook_path = self.hooks_dir / hook_name
            if hook_path.exists() and self._is_our_hook(hook_path):
                hooks[hook_name] = True

        return hooks

    def _is_our_hook(self, hook_path: Path) -> bool:
        """
        Check if a hook file is one of our hooks.

        Args:
            hook_path: Path to the hook file.

        Returns:
            True if the hook is one of our hooks, False otherwise.
        """
        try:
            with open(hook_path, "r") as f:
                content = f.read()
                return "embedding pipeline" in content and "python -m src.git_hooks.trigger" in content
        except (IOError, OSError, FileNotFoundError):
            return False


def install_hooks(repo_path: Optional[str] = None) -> List[str]:
    """
    Install all hooks for the embedding pipeline.

    Args:
        repo_path: Path to the Git repository. If None, the current directory is used.

    Returns:
        List of paths to the installed hook files.
    """
    installer = GitHookInstaller(repo_path)
    return installer.install_all_hooks()


def check_hooks_status(repo_path: Optional[str] = None) -> dict:
    """
    Check the status of the hooks.

    Args:
        repo_path: Path to the Git repository. If None, the current directory is used.

    Returns:
        Dictionary with hook names as keys and boolean indicating if they are installed.
    """
    installer = GitHookInstaller(repo_path)
    return installer.check_hook_status()