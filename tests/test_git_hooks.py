#!/usr/bin/env python3
"""
Test script for the Git hook functionality.

This script tests the Git integration functionality by creating a temporary
repository and simulating various Git operations.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.git_hooks.change_detector import GitChangeDetector
from src.git_hooks.installer import GitHookInstaller


class GitHooksTestCase(unittest.TestCase):
    """Test case for the Git hooks functionality."""

    def setUp(self):
        """Set up a temporary Git repository for testing."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = os.path.join(self.test_dir, "test-repo")

        # Initialize a Git repository
        os.makedirs(self.repo_path)
        subprocess.run(["git", "init"], cwd=self.repo_path, check=True)

        # Configure Git user name and email (required for commits)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo_path, check=True)

        # Create some initial files
        self._create_file("test1.md", "# Test 1\n\nThis is a test file.")
        self._create_file("test2.md", "# Test 2\n\nThis is another test file.")
        self._create_file("not-markdown.txt", "This is not a markdown file.")

        # Commit the initial files
        subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.repo_path, check=True)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def _create_file(self, filename, content):
        """Create a file in the test repository."""
        file_path = os.path.join(self.repo_path, filename)
        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    def _modify_file(self, filename, content):
        """Modify a file in the test repository."""
        file_path = os.path.join(self.repo_path, filename)
        with open(file_path, "a") as f:
            f.write(content)
        return file_path

    def _delete_file(self, filename):
        """Delete a file in the test repository."""
        file_path = os.path.join(self.repo_path, filename)
        os.remove(file_path)
        return file_path

    def test_change_detector_initialization(self):
        """Test initializing the GitChangeDetector."""
        detector = GitChangeDetector(self.repo_path)
        self.assertIsNotNone(detector)

    def test_hook_installer_initialization(self):
        """Test initializing the GitHookInstaller."""
        installer = GitHookInstaller(self.repo_path)
        self.assertIsNotNone(installer)
        self.assertTrue(os.path.exists(installer.hooks_dir))

    def test_get_all_markdown_files(self):
        """Test getting all markdown files tracked by Git."""
        detector = GitChangeDetector(self.repo_path)
        files = detector.get_all_markdown_files()

        # Check that only markdown files are returned
        self.assertEqual(len(files), 2)
        self.assertTrue(any(f.endswith("test1.md") for f in files))
        self.assertTrue(any(f.endswith("test2.md") for f in files))
        self.assertFalse(any(f.endswith("not-markdown.txt") for f in files))

    def test_detect_added_file(self):
        """Test detecting an added file."""
        # Add a new file
        self._create_file("test3.md", "# Test 3\n\nThis is a new test file.")
        subprocess.run(["git", "add", "test3.md"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Add test3.md"], cwd=self.repo_path, check=True)

        # Detect changes
        detector = GitChangeDetector(self.repo_path)
        changes = detector.get_last_commit_changes()

        # Check that the added file is detected
        self.assertEqual(len(changes["added"]), 1)
        self.assertTrue(any(f.endswith("test3.md") for f in changes["added"]))
        self.assertEqual(len(changes["modified"]), 0)
        self.assertEqual(len(changes["deleted"]), 0)

    def test_detect_modified_file(self):
        """Test detecting a modified file."""
        # Modify a file
        self._modify_file("test1.md", "\n\nThis is an update to test1.md.")
        subprocess.run(["git", "add", "test1.md"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Modify test1.md"], cwd=self.repo_path, check=True)

        # Detect changes
        detector = GitChangeDetector(self.repo_path)
        changes = detector.get_last_commit_changes()

        # Check that the modified file is detected
        self.assertEqual(len(changes["modified"]), 1)
        self.assertTrue(any(f.endswith("test1.md") for f in changes["modified"]))
        self.assertEqual(len(changes["added"]), 0)
        self.assertEqual(len(changes["deleted"]), 0)

    def test_detect_deleted_file(self):
        """Test detecting a deleted file."""
        # Delete a file
        self._delete_file("test2.md")
        subprocess.run(["git", "add", "--all"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Delete test2.md"], cwd=self.repo_path, check=True)

        # Detect changes
        detector = GitChangeDetector(self.repo_path)
        changes = detector.get_last_commit_changes()

        # Check that the deleted file is detected
        self.assertEqual(len(changes["deleted"]), 1)
        self.assertTrue(any(f.endswith("test2.md") for f in changes["deleted"]))
        self.assertEqual(len(changes["added"]), 0)
        self.assertEqual(len(changes["modified"]), 0)

    def test_install_hooks(self):
        """Test installing Git hooks."""
        installer = GitHookInstaller(self.repo_path)
        installed_hooks = installer.install_all_hooks()

        # Check that the hooks are installed
        self.assertEqual(len(installed_hooks), 2)

        # Check that the hook files exist
        hook_dir = Path(self.repo_path) / ".git" / "hooks"
        self.assertTrue((hook_dir / "post-commit").exists())
        self.assertTrue((hook_dir / "post-merge").exists())

        # Check that the hook files are executable
        self.assertTrue(os.access(hook_dir / "post-commit", os.X_OK))
        self.assertTrue(os.access(hook_dir / "post-merge", os.X_OK))

        # Check the hook status
        hook_status = installer.check_hook_status()
        self.assertTrue(hook_status["post-commit"])
        self.assertTrue(hook_status["post-merge"])


if __name__ == "__main__":
    unittest.main()