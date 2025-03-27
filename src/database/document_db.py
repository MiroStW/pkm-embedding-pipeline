#!/usr/bin/env python3
"""
Document tracking database.

This module provides functionality for tracking document processing status.
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class DocumentTracker:
    """Tracks document processing status for the embedding pipeline."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the DocumentTracker.

        Args:
            db_path: Path to the SQLite database file. If None, a default path is used.
        """
        if db_path is None:
            # Use default path in the project directory
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(root_dir, "data", "document_tracker.db")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create documents table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            path TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            last_modified TIMESTAMP,
            last_processed TIMESTAMP,
            error TEXT
        )
        ''')

        conn.commit()
        conn.close()

    def mark_for_processing(self, file_path: str) -> bool:
        """
        Mark a document for processing.

        Args:
            file_path: Path to the document.

        Returns:
            True if successful, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if the document already exists
            cursor.execute("SELECT path FROM documents WHERE path = ?", (file_path,))
            result = cursor.fetchone()

            if result:
                # Update the existing record
                cursor.execute(
                    "UPDATE documents SET status = 'pending', last_modified = CURRENT_TIMESTAMP, error = NULL WHERE path = ?",
                    (file_path,)
                )
            else:
                # Insert a new record
                cursor.execute(
                    "INSERT INTO documents (path, status, last_modified) VALUES (?, 'pending', CURRENT_TIMESTAMP)",
                    (file_path,)
                )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error marking document for processing: {e}")
            return False

    def mark_for_deletion(self, file_path: str) -> bool:
        """
        Mark a document for deletion from the vector database.

        Args:
            file_path: Path to the document.

        Returns:
            True if successful, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if the document already exists
            cursor.execute("SELECT path FROM documents WHERE path = ?", (file_path,))
            result = cursor.fetchone()

            if result:
                # Update the existing record
                cursor.execute(
                    "UPDATE documents SET status = 'delete', last_modified = CURRENT_TIMESTAMP WHERE path = ?",
                    (file_path,)
                )
            else:
                # Insert a new record
                cursor.execute(
                    "INSERT INTO documents (path, status, last_modified) VALUES (?, 'delete', CURRENT_TIMESTAMP)",
                    (file_path,)
                )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error marking document for deletion: {e}")
            return False

    def get_files_for_processing(self) -> List[str]:
        """
        Get files marked for processing.

        Returns:
            List of file paths marked for processing.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT path FROM documents WHERE status = 'pending'")
            files = [row[0] for row in cursor.fetchall()]

            conn.close()
            return files
        except Exception as e:
            logger.error(f"Error getting files for processing: {e}")
            return []

    def get_files_for_deletion(self) -> List[str]:
        """
        Get files marked for deletion.

        Returns:
            List of file paths marked for deletion.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT path FROM documents WHERE status = 'delete'")
            files = [row[0] for row in cursor.fetchall()]

            conn.close()
            return files
        except Exception as e:
            logger.error(f"Error getting files for deletion: {e}")
            return []