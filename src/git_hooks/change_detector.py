#!/usr/bin/env python3
"""
Change detection module for Git integration.

This module provides functionality to detect changes in Git repositories,
including added, modified, and deleted files.
"""

import os
import subprocess
from typing import List, Dict, Tuple, Set


class GitChangeDetector:
    """Detects changes in Git repositories and tracks modified files."""

    def __init__(self, repo_path: str = None):
        """
        Initialize the GitChangeDetector.

        Args:
            repo_path: Path to the Git repository. If None, the current directory is used.
        """
        self.repo_path = repo_path or os.getcwd()
        self._ensure_git_repo()

    def _ensure_git_repo(self):
        """Ensure that the repo_path is a Git repository."""
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError:
            raise ValueError(f"Not a Git repository: {self.repo_path}")

    def _run_git_command(self, command: List[str]) -> str:
        """
        Run a Git command and return its output.

        Args:
            command: Git command as a list of strings.

        Returns:
            Output of the Git command as a string.
        """
        result = subprocess.run(
            command,
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return result.stdout.strip()

    def get_last_commit_changes(self) -> Dict[str, List[str]]:
        """
        Get files changed in the last commit.

        Returns:
            Dictionary with keys 'added', 'modified', and 'deleted', each containing a list of file paths.
        """
        # Get the last commit hash
        last_commit = self._run_git_command(["git", "rev-parse", "HEAD"])

        # Get the parent commit hash (for comparison)
        parent_commit = self._run_git_command(["git", "rev-parse", "HEAD~1"])

        # Get changes between the last commit and its parent
        return self.get_changes_between_commits(parent_commit, last_commit)

    def get_merge_changes(self) -> Dict[str, List[str]]:
        """
        Get files changed in the last merge or pull.

        Returns:
            Dictionary with keys 'added', 'modified', and 'deleted', each containing a list of file paths.
        """
        # ORIG_HEAD points to the previous state before a merge or pull
        try:
            orig_head = self._run_git_command(["git", "rev-parse", "ORIG_HEAD"])
            head = self._run_git_command(["git", "rev-parse", "HEAD"])
            return self.get_changes_between_commits(orig_head, head)
        except subprocess.CalledProcessError:
            # If ORIG_HEAD doesn't exist, fall back to comparing with the last commit
            return self.get_last_commit_changes()

    def get_changes_between_commits(self, old_commit: str, new_commit: str) -> Dict[str, List[str]]:
        """
        Get changes between two commits.

        Args:
            old_commit: Hash of the older commit.
            new_commit: Hash of the newer commit.

        Returns:
            Dictionary with keys 'added', 'modified', and 'deleted', each containing a list of file paths.
        """
        result = {
            'added': [],
            'modified': [],
            'deleted': []
        }

        # Get the diff between the two commits
        diff_output = self._run_git_command([
            "git", "diff", "--name-status", old_commit, new_commit
        ])

        if not diff_output:
            return result

        # Parse the diff output
        for line in diff_output.splitlines():
            parts = line.split('\t')
            if len(parts) < 2:
                continue

            status, file_path = parts[0], parts[1]

            # Check if it's a markdown file
            if not file_path.endswith('.md'):
                continue

            if status.startswith('A'):
                result['added'].append(file_path)
            elif status.startswith('M'):
                result['modified'].append(file_path)
            elif status.startswith('D'):
                result['deleted'].append(file_path)
            # Handle renamed files (R)
            elif status.startswith('R'):
                if len(parts) >= 3:
                    old_path, new_path = parts[1], parts[2]
                    if old_path.endswith('.md'):
                        result['deleted'].append(old_path)
                    if new_path.endswith('.md'):
                        result['added'].append(new_path)

        return result

    def get_all_markdown_files(self) -> List[str]:
        """
        Get all markdown files tracked by Git.

        Returns:
            List of paths to all markdown files in the repository.
        """
        output = self._run_git_command([
            "git", "ls-files", "*.md"
        ])
        return output.splitlines() if output else []