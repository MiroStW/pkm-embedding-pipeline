"""
Database manager module for tracking document states and managing work queue.
"""
import datetime
from datetime import timezone
import logging
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc

from src.database.init_db import Document, ProcessingQueue, init_db

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages document tracking and persistent work queue operations.
    """

    def __init__(self, test_db_path=None):
        """
        Initialize the database manager with a session factory.

        Args:
            test_db_path: Path to test database file (for testing only)
        """
        # Check if TEST_DB_PATH is set in environment
        env_test_path = os.environ.get("TEST_DB_PATH")

        # Use test path if provided directly or through environment
        if test_db_path or env_test_path:
            db_path = test_db_path or env_test_path
            self.engine, self.SessionFactory = init_db(test_mode=True, db_path=db_path)
        else:
            self.engine, self.SessionFactory = init_db()

    def get_session(self) -> Session:
        """Create and return a new database session."""
        return self.SessionFactory()

    # Document state tracking methods

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.

        Args:
            document_id: The unique ID of the document

        Returns:
            Document data as a dictionary or None if not found
        """
        with self.get_session() as session:
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                return {
                    'id': document.id,
                    'filepath': document.filepath,
                    'title': document.title,
                    'last_modified': document.last_modified,
                    'last_processed': document.last_processed,
                    'status': document.status,
                    'error_message': document.error_message,
                    'chunk_count': document.chunk_count,
                    'embedding_model': document.embedding_model,
                    'hash': document.hash
                }
            return None

    def get_document_by_filepath(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its filepath.

        Args:
            filepath: The filepath of the document

        Returns:
            Document data as a dictionary or None if not found
        """
        with self.get_session() as session:
            document = session.query(Document).filter(Document.filepath == filepath).first()
            if document:
                return {
                    'id': document.id,
                    'filepath': document.filepath,
                    'title': document.title,
                    'last_modified': document.last_modified,
                    'last_processed': document.last_processed,
                    'status': document.status,
                    'error_message': document.error_message,
                    'chunk_count': document.chunk_count,
                    'embedding_model': document.embedding_model,
                    'hash': document.hash
                }
            return None

    def create_or_update_document(self,
                                  document_id: str,
                                  filepath: str,
                                  title: Optional[str] = None,
                                  last_modified: Optional[datetime.datetime] = None,
                                  status: str = "pending",
                                  hash_value: Optional[str] = None) -> str:
        """
        Create a new document record or update an existing one.

        Args:
            document_id: The unique ID of the document
            filepath: The path to the document file
            title: The document title
            last_modified: The last modification timestamp
            status: Document processing status (pending, processing, completed, error)
            hash_value: Content hash for change detection

        Returns:
            The document ID
        """
        try:
            with self.get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()

                if document:
                    # Update existing record
                    if filepath:
                        document.filepath = filepath
                    if title is not None:
                        document.title = title
                    if last_modified is not None:
                        document.last_modified = last_modified
                    if status:
                        document.status = status
                    if hash_value:
                        document.hash = hash_value
                else:
                    # Create new record
                    document = Document(
                        id=document_id,
                        filepath=filepath,
                        title=title,
                        last_modified=last_modified or datetime.datetime.now(timezone.utc),
                        status=status,
                        hash=hash_value
                    )
                    session.add(document)

                session.commit()
                return document_id

        except SQLAlchemyError as e:
            logger.error(f"Database error while creating/updating document {document_id}: {str(e)}")
            raise

    def update_document_status(self,
                               document_id: str,
                               status: str,
                               error_message: Optional[str] = None,
                               chunk_count: Optional[int] = None,
                               embedding_model: Optional[str] = None) -> bool:
        """
        Update the processing status of a document.

        Args:
            document_id: The unique ID of the document
            status: New status (pending, processing, completed, error)
            error_message: Error message if status is 'error'
            chunk_count: Number of chunks created from the document
            embedding_model: Name of the embedding model used

        Returns:
            True if the update was successful, False otherwise
        """
        try:
            with self.get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                if not document:
                    logger.warning(f"Attempted to update status for non-existent document {document_id}")
                    return False

                document.status = status
                if status == 'completed' or status == 'error':
                    document.last_processed = datetime.datetime.now(timezone.utc)

                if error_message is not None:
                    document.error_message = error_message

                if chunk_count is not None:
                    document.chunk_count = chunk_count

                if embedding_model is not None:
                    document.embedding_model = embedding_model

                session.commit()
                return True

        except SQLAlchemyError as e:
            logger.error(f"Database error while updating document status for {document_id}: {str(e)}")
            return False

    def calculate_content_hash(self, content: str) -> str:
        """
        Calculate a hash of the document content for change detection.

        Args:
            content: Document content

        Returns:
            Hash string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def is_document_changed(self, document_id: str, content: str) -> bool:
        """
        Check if a document's content has changed by comparing hash values.

        Args:
            document_id: The unique ID of the document
            content: Current document content

        Returns:
            True if the document has changed, False otherwise
        """
        current_hash = self.calculate_content_hash(content)
        document = self.get_document(document_id)

        if not document or not document.get('hash'):
            return True

        return document['hash'] != current_hash

    def get_documents_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of document dictionaries
        """
        with self.get_session() as session:
            documents = session.query(Document).filter(Document.status == status).all()
            return [
                {
                    'id': doc.id,
                    'filepath': doc.filepath,
                    'title': doc.title,
                    'last_modified': doc.last_modified,
                    'last_processed': doc.last_processed,
                    'status': doc.status,
                    'error_message': doc.error_message,
                    'chunk_count': doc.chunk_count,
                    'embedding_model': doc.embedding_model,
                    'hash': doc.hash
                }
                for doc in documents
            ]

    # Work queue methods

    def enqueue_document(self, document_id: str, priority: int = 0) -> int:
        """
        Add a document to the processing queue.

        Args:
            document_id: The unique ID of the document to process
            priority: Processing priority (higher numbers = higher priority)

        Returns:
            Queue item ID
        """
        try:
            with self.get_session() as session:
                # Check if document is already in queue
                existing_item = session.query(ProcessingQueue).filter(
                    ProcessingQueue.document_id == document_id,
                    ProcessingQueue.is_completed == False
                ).first()

                if existing_item:
                    # Update priority if needed
                    if existing_item.priority < priority:
                        existing_item.priority = priority
                        session.commit()
                    return existing_item.id

                # Otherwise create new queue item
                queue_item = ProcessingQueue(
                    document_id=document_id,
                    priority=priority,
                    created_at=datetime.datetime.now(timezone.utc)
                )
                session.add(queue_item)
                session.commit()

                # Update document status
                self.update_document_status(document_id, "pending")

                return queue_item.id

        except SQLAlchemyError as e:
            logger.error(f"Database error while enqueueing document {document_id}: {str(e)}")
            raise

    def dequeue_document(self, count: int = 1) -> List[Tuple[int, str]]:
        """
        Get the next document(s) from the processing queue.

        Args:
            count: Number of queue items to retrieve

        Returns:
            List of tuples (queue_item_id, document_id)
        """
        try:
            with self.get_session() as session:
                # Get items ordered by priority (higher first) then creation time
                queue_items = session.query(ProcessingQueue).filter(
                    ProcessingQueue.is_completed == False,
                    ProcessingQueue.processing_started == None
                ).order_by(
                    desc(ProcessingQueue.priority),
                    ProcessingQueue.created_at
                ).limit(count).all()

                results = []
                for item in queue_items:
                    # Mark as processing
                    item.processing_started = datetime.datetime.now(timezone.utc)
                    results.append((item.id, item.document_id))

                    # Update document status
                    document = session.query(Document).filter(Document.id == item.document_id).first()
                    if document:
                        document.status = "processing"

                session.commit()
                return results

        except SQLAlchemyError as e:
            logger.error(f"Database error while dequeuing documents: {str(e)}")
            return []

    def complete_queue_item(self, queue_item_id: int, success: bool) -> bool:
        """
        Mark a queue item as completed.

        Args:
            queue_item_id: ID of the queue item
            success: Whether processing was successful

        Returns:
            True if the update was successful, False otherwise
        """
        try:
            with self.get_session() as session:
                queue_item = session.query(ProcessingQueue).filter(ProcessingQueue.id == queue_item_id).first()
                if not queue_item:
                    logger.warning(f"Attempted to complete non-existent queue item {queue_item_id}")
                    return False

                queue_item.is_completed = True
                session.commit()
                return True

        except SQLAlchemyError as e:
            logger.error(f"Database error while completing queue item {queue_item_id}: {str(e)}")
            return False

    def reset_stalled_queue_items(self, time_threshold_minutes: int = 30) -> int:
        """
        Reset queue items that started processing but never completed.

        Args:
            time_threshold_minutes: Time in minutes after which a processing item is considered stalled

        Returns:
            Number of reset items
        """
        threshold_time = datetime.datetime.now(timezone.utc) - datetime.timedelta(minutes=time_threshold_minutes)

        try:
            with self.get_session() as session:
                # Find stalled items
                stalled_items = session.query(ProcessingQueue).filter(
                    ProcessingQueue.is_completed == False,
                    ProcessingQueue.processing_started != None,
                    ProcessingQueue.processing_started < threshold_time
                ).all()

                # Reset them
                for item in stalled_items:
                    item.processing_started = None

                    # Also reset document status
                    document = session.query(Document).filter(Document.id == item.document_id).first()
                    if document and document.status == "processing":
                        document.status = "pending"

                session.commit()
                return len(stalled_items)

        except SQLAlchemyError as e:
            logger.error(f"Database error while resetting stalled queue items: {str(e)}")
            return 0

    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get statistics about the processing queue.

        Returns:
            Dictionary with queue statistics
        """
        try:
            with self.get_session() as session:
                pending = session.query(ProcessingQueue).filter(
                    ProcessingQueue.is_completed == False,
                    ProcessingQueue.processing_started == None
                ).count()

                processing = session.query(ProcessingQueue).filter(
                    ProcessingQueue.is_completed == False,
                    ProcessingQueue.processing_started != None
                ).count()

                completed = session.query(ProcessingQueue).filter(
                    ProcessingQueue.is_completed == True
                ).count()

                return {
                    'pending': pending,
                    'processing': processing,
                    'completed': completed,
                    'total': pending + processing + completed
                }

        except SQLAlchemyError as e:
            logger.error(f"Database error while getting queue stats: {str(e)}")
            return {'pending': 0, 'processing': 0, 'completed': 0, 'total': 0}