#!/usr/bin/env python3
"""
Document tracking database.

This module provides functionality for tracking document processing status.
"""

import os
import sqlite3
import logging
import json
from typing import List, Dict, Optional, Any

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
            metadata TEXT,
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

    def is_processed(self, file_path: str) -> bool:
        """
        Check if a document has been successfully processed.

        Args:
            file_path: Path to the document.

        Returns:
            True if the document has been successfully processed, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT status FROM documents WHERE path = ?", (file_path,))
            result = cursor.fetchone()

            conn.close()

            # Return True if the document exists and has status 'completed'
            return result is not None and result[0] == 'completed'
        except Exception as e:
            logger.error(f"Error checking if document is processed: {e}")
            return False

    def mark_completed(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Mark a document as successfully processed.

        Args:
            file_path: Path to the document.
            metadata: Document metadata to store.

        Returns:
            True if successful, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Serialize metadata to JSON
            metadata_json = json.dumps(metadata)

            # Update the document status
            cursor.execute(
                "UPDATE documents SET status = 'completed', last_processed = CURRENT_TIMESTAMP, metadata = ? WHERE path = ?",
                (metadata_json, file_path)
            )

            # If the document doesn't exist, insert it
            if cursor.rowcount == 0:
                cursor.execute(
                    "INSERT INTO documents (path, status, last_modified, last_processed, metadata) VALUES (?, 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)",
                    (file_path, metadata_json)
                )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error marking document as completed: {e}")
            return False

    def mark_error(self, file_path: str, error_message: str) -> bool:
        """
        Mark a document as having an error during processing.

        Args:
            file_path: Path to the document.
            error_message: Error message to store.

        Returns:
            True if successful, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Update the document status
            cursor.execute(
                "UPDATE documents SET status = 'error', last_processed = CURRENT_TIMESTAMP, error = ? WHERE path = ?",
                (error_message, file_path)
            )

            # If the document doesn't exist, insert it
            if cursor.rowcount == 0:
                cursor.execute(
                    "INSERT INTO documents (path, status, last_modified, last_processed, error) VALUES (?, 'error', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)",
                    (file_path, error_message)
                )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error marking document with error: {e}")
            return False

    def get_all_files(self) -> List[str]:
        """
        Get all files tracked in the database.

        Returns:
            List of all file paths in the database.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT path FROM documents")
            files = [row[0] for row in cursor.fetchall()]

            conn.close()
            return files
        except Exception as e:
            logger.error(f"Error getting all files: {e}")
            return []