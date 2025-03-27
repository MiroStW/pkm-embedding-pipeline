#!/usr/bin/env python3
"""
Document tracking database.

This module provides functionality for tracking document processing status.
"""

import os
import sqlite3
import logging
import json
import time
from typing import List, Dict, Optional, Any, Tuple

from src.database.init_db import init_db_sqlite

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
        try:
            # Use the improved database initialization
            conn, cursor = init_db_sqlite()
            conn.close()
            logger.debug("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            # Fallback to basic initialization if the improved one fails
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
            metadata: Document metadata.

        Returns:
            True if successful, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Serialize metadata to JSON
            metadata_json = json.dumps(metadata)

            # Check if the document already exists
            cursor.execute("SELECT path FROM documents WHERE path = ?", (file_path,))
            result = cursor.fetchone()

            if result:
                # Update the existing record
                cursor.execute(
                    "UPDATE documents SET status = 'completed', last_processed = CURRENT_TIMESTAMP, metadata = ? WHERE path = ?",
                    (metadata_json, file_path)
                )
            else:
                # Insert a new record
                cursor.execute(
                    "INSERT INTO documents (path, status, last_processed, metadata) VALUES (?, 'completed', CURRENT_TIMESTAMP, ?)",
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
        Mark a document as failed processing.

        Args:
            file_path: Path to the document.
            error_message: Error message describing why processing failed.

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
                    "UPDATE documents SET status = 'error', last_processed = CURRENT_TIMESTAMP, error = ? WHERE path = ?",
                    (error_message, file_path)
                )
            else:
                # Insert a new record
                cursor.execute(
                    "INSERT INTO documents (path, status, last_processed, error) VALUES (?, 'error', CURRENT_TIMESTAMP, ?)",
                    (file_path, error_message)
                )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error marking document as error: {e}")
            return False

    def get_all_files(self) -> List[str]:
        """
        Get all files tracked by the document tracker.

        Returns:
            List of all tracked file paths.
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

    def reset_file(self, file_path: str) -> bool:
        """
        Reset a file's status back to pending for reprocessing.

        Args:
            file_path: Path to the document.

        Returns:
            True if successful, False otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if the document exists
            cursor.execute("SELECT path FROM documents WHERE path = ?", (file_path,))
            result = cursor.fetchone()

            if result:
                # Update the existing record to pending
                cursor.execute(
                    "UPDATE documents SET status = 'pending', last_modified = CURRENT_TIMESTAMP, error = NULL WHERE path = ?",
                    (file_path,)
                )
                conn.commit()
                conn.close()
                return True
            else:
                # File not tracked yet, mark for processing instead
                conn.close()
                return self.mark_for_processing(file_path)
        except Exception as e:
            logger.error(f"Error resetting file status: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about document tracking status.

        Returns:
            Dictionary with statistics about tracked documents.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM documents")
            total = cursor.fetchone()[0]

            # Get count by status
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'completed'")
            completed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'error'")
            errors = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'pending'")
            pending = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'delete'")
            to_delete = cursor.fetchone()[0]

            conn.close()

            # Calculate percentages
            completed_percentage = (completed / total * 100) if total > 0 else 0
            error_percentage = (errors / total * 100) if total > 0 else 0
            pending_percentage = (pending / total * 100) if total > 0 else 0
            delete_percentage = (to_delete / total * 100) if total > 0 else 0

            return {
                'total': total,
                'completed': completed,
                'completed_percentage': completed_percentage,
                'errors': errors,
                'error_percentage': error_percentage,
                'pending': pending,
                'pending_percentage': pending_percentage,
                'to_delete': to_delete,
                'delete_percentage': delete_percentage
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'total': 0,
                'completed': 0,
                'completed_percentage': 0,
                'errors': 0,
                'error_percentage': 0,
                'pending': 0,
                'pending_percentage': 0,
                'to_delete': 0,
                'delete_percentage': 0
            }

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents with their status and metadata.

        Returns:
            List of dictionaries with document information.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT path, status, last_modified, last_processed, metadata, error
                FROM documents
            """)
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                file_path, status, last_modified, last_processed, metadata_json, error = row

                # Parse timestamps and metadata
                # Convert to timestamp if exists, otherwise use 0
                timestamp = time.time()
                last_modified_ts = last_modified or timestamp
                last_processed_ts = last_processed or timestamp

                # Parse metadata if it exists
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}

                documents.append({
                    'file_path': file_path,
                    'status': status,
                    'timestamp': last_processed_ts or last_modified_ts,
                    'last_modified': last_modified_ts,
                    'last_processed': last_processed_ts,
                    'metadata': metadata,
                    'error': error
                })

            conn.close()
            return documents
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    def get_completed_documents(self) -> List[Dict[str, Any]]:
        """
        Get all completed documents.

        Returns:
            List of dictionaries with document information.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT path, status, last_modified, last_processed, metadata, error
                FROM documents
                WHERE status = 'completed'
            """)
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                file_path, status, last_modified, last_processed, metadata_json, error = row

                # Parse timestamps and metadata
                timestamp = time.time()
                last_modified_ts = last_modified or timestamp
                last_processed_ts = last_processed or timestamp

                # Parse metadata if it exists
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}

                documents.append({
                    'file_path': file_path,
                    'status': status,
                    'timestamp': last_processed_ts,
                    'last_modified': last_modified_ts,
                    'last_processed': last_processed_ts,
                    'metadata': metadata,
                    'error': error
                })

            conn.close()
            return documents
        except Exception as e:
            logger.error(f"Error getting completed documents: {e}")
            return []

    def get_error_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents with errors.

        Returns:
            List of dictionaries with document information.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT path, status, last_modified, last_processed, metadata, error
                FROM documents
                WHERE status = 'error'
            """)
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                file_path, status, last_modified, last_processed, metadata_json, error = row

                # Parse timestamps and metadata
                timestamp = time.time()
                last_modified_ts = last_modified or timestamp
                last_processed_ts = last_processed or timestamp

                # Parse metadata if it exists
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}

                documents.append({
                    'file_path': file_path,
                    'status': status,
                    'timestamp': last_processed_ts or last_modified_ts,
                    'last_modified': last_modified_ts,
                    'last_processed': last_processed_ts,
                    'metadata': metadata,
                    'error': error
                })

            conn.close()
            return documents
        except Exception as e:
            logger.error(f"Error getting error documents: {e}")
            return []

    def get_pending_documents(self) -> List[Dict[str, Any]]:
        """
        Get all pending documents.

        Returns:
            List of dictionaries with document information.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT path, status, last_modified, last_processed, metadata, error
                FROM documents
                WHERE status = 'pending'
            """)
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                file_path, status, last_modified, last_processed, metadata_json, error = row

                # Parse timestamps and metadata
                timestamp = time.time()
                last_modified_ts = last_modified or timestamp
                last_processed_ts = last_processed or timestamp

                # Parse metadata if it exists
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}

                documents.append({
                    'file_path': file_path,
                    'status': status,
                    'timestamp': last_modified_ts,
                    'last_modified': last_modified_ts,
                    'last_processed': last_processed_ts,
                    'metadata': metadata,
                    'error': error
                })

            conn.close()
            return documents
        except Exception as e:
            logger.error(f"Error getting pending documents: {e}")
            return []

    def find_inconsistencies(self) -> List[str]:
        """
        Find inconsistencies in the document database.

        Returns:
            List of inconsistency descriptions.
        """
        inconsistencies = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check for invalid status values
            cursor.execute("SELECT path, status FROM documents WHERE status NOT IN ('pending', 'completed', 'error', 'delete')")
            invalid_status = cursor.fetchall()
            if invalid_status:
                for path, status in invalid_status:
                    inconsistencies.append(f"Invalid status '{status}' for file: {path}")

            # Check for files marked as completed but with no metadata
            cursor.execute("SELECT path FROM documents WHERE status = 'completed' AND (metadata IS NULL OR metadata = '')")
            no_metadata = cursor.fetchall()
            if no_metadata:
                for (path,) in no_metadata:
                    inconsistencies.append(f"Completed file missing metadata: {path}")

            # Check for files marked as error but with no error message
            cursor.execute("SELECT path FROM documents WHERE status = 'error' AND (error IS NULL OR error = '')")
            no_error = cursor.fetchall()
            if no_error:
                for (path,) in no_error:
                    inconsistencies.append(f"Error file missing error message: {path}")

            conn.close()
            return inconsistencies

        except Exception as e:
            logger.error(f"Error finding inconsistencies: {e}")
            inconsistencies.append(f"Error accessing database: {str(e)}")
            return inconsistencies

    def repair_inconsistencies(self) -> List[str]:
        """
        Repair inconsistencies in the document database.

        Returns:
            List of applied fixes.
        """
        fixes = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Fix invalid status values
            cursor.execute("UPDATE documents SET status = 'pending' WHERE status NOT IN ('pending', 'completed', 'error', 'delete')")
            if cursor.rowcount > 0:
                fixes.append(f"Reset {cursor.rowcount} documents with invalid status to 'pending'")

            # Fix completed files without metadata
            cursor.execute("UPDATE documents SET metadata = '{}' WHERE status = 'completed' AND (metadata IS NULL OR metadata = '')")
            if cursor.rowcount > 0:
                fixes.append(f"Added empty metadata to {cursor.rowcount} completed documents")

            # Fix error files without error message
            cursor.execute("UPDATE documents SET error = 'Unknown error' WHERE status = 'error' AND (error IS NULL OR error = '')")
            if cursor.rowcount > 0:
                fixes.append(f"Added default error message to {cursor.rowcount} error documents")

            conn.commit()
            conn.close()
            return fixes

        except Exception as e:
            logger.error(f"Error repairing inconsistencies: {e}")
            fixes.append(f"Failed to repair: {str(e)}")
            return fixes