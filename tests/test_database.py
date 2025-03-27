"""
Tests for database components: document tracking, queue persistence, and checkpoints.
"""
import sys
import time
import uuid
import logging
import unittest
from datetime import datetime
import tempfile
import os
import shutil

# Make sure we can import from src
sys.path.insert(0, '.')

from src.database.db_manager import DatabaseManager
from src.database.checkpoint import CheckpointManager
from src.database.init_db import init_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDatabase(unittest.TestCase):
    """Test database components for the embedding pipeline."""

    def setUp(self):
        """Set up a test database environment"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_database.db")

        # Override the database path for testing
        # This will ensure we don't use the real database for tests
        os.environ["TEST_DB_PATH"] = self.test_db_path

        # Run database initialization to create schema
        # The init_db function reads TEST_DB_PATH if it exists
        self.engine, self.Session = init_db(test_mode=True, db_path=self.test_db_path)

        logger.info(f"Test database initialized at {self.test_db_path}")

    def tearDown(self):
        """Clean up test environment"""
        # Delete test database
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Remove environment variable
        if "TEST_DB_PATH" in os.environ:
            del os.environ["TEST_DB_PATH"]

        logger.info("Test database cleaned up")

    def test_document_tracking(self):
        """Test document state tracking."""
        logger.info("=== Testing Document State Tracking ===")

        # Create a database manager
        db_manager = DatabaseManager()

        # Generate a test document ID
        doc_id = f"test_doc_{uuid.uuid4().hex[:8]}"
        filepath = f"/test/path/document_{doc_id}.md"

        logger.info(f"Creating test document {doc_id}")

        # Create a document
        db_manager.create_or_update_document(
            document_id=doc_id,
            filepath=filepath,
            title="Test Document",
            last_modified=datetime.utcnow(),
            status="pending",
            hash_value="initial_hash_value"
        )

        # Retrieve and verify document
        doc = db_manager.get_document(doc_id)
        self.assertIsNotNone(doc, "Failed to retrieve created document")
        self.assertEqual(doc['id'], doc_id)
        self.assertEqual(doc['status'], "pending")
        logger.info(f"Successfully created document: {doc['id']} with status: {doc['status']}")

        # Update document status
        logger.info(f"Updating document status to 'processing'")
        success = db_manager.update_document_status(
            document_id=doc_id,
            status="processing"
        )

        self.assertTrue(success, "Failed to update document status")

        # Verify status changed
        doc = db_manager.get_document(doc_id)
        self.assertEqual(doc['status'], "processing", "Document status not updated correctly")
        logger.info("Successfully updated document status to 'processing'")

        # Test hash-based change detection
        test_content = "This is test content for our document."
        hash_value = db_manager.calculate_content_hash(test_content)

        # Update the document hash
        db_manager.create_or_update_document(
            document_id=doc_id,
            filepath=filepath,
            hash_value=hash_value
        )

        # Check if document has changed
        changed = db_manager.is_document_changed(doc_id, test_content)
        self.assertFalse(changed, "Document incorrectly detected as changed when content is the same")
        logger.info("Document change detection working correctly for unchanged content")

        # Check if document has changed with different content
        changed = db_manager.is_document_changed(doc_id, test_content + " Modified!")
        self.assertTrue(changed, "Document change not detected when content is different")
        logger.info("Document change detection working correctly for changed content")

        # Complete the document processing
        logger.info(f"Completing document processing with status 'completed'")
        success = db_manager.update_document_status(
            document_id=doc_id,
            status="completed",
            chunk_count=5,
            embedding_model="test-model"
        )

        self.assertTrue(success, "Failed to complete document processing")

        # Verify completion
        doc = db_manager.get_document(doc_id)
        self.assertEqual(doc['status'], "completed", "Document status not updated to completed")
        self.assertEqual(doc['chunk_count'], 5, "Document chunk count not updated")
        self.assertEqual(doc['embedding_model'], "test-model", "Document embedding model not updated")

        logger.info("Document state tracking test passed")

    def test_queue_persistence(self):
        """Test processing queue persistence through restarts."""
        logger.info("=== Testing Processing Queue Persistence ===")

        # Create a database manager
        db_manager = DatabaseManager()

        # Generate test document IDs
        docs = []
        for i in range(3):
            doc_id = f"queue_test_doc_{uuid.uuid4().hex[:8]}"
            filepath = f"/test/path/queue_document_{i}.md"

            # Create document
            db_manager.create_or_update_document(
                document_id=doc_id,
                filepath=filepath,
                title=f"Queue Test Document {i}",
                status="pending"
            )
            docs.append(doc_id)

        # Add documents to processing queue with different priorities
        queue_ids = []
        for i, doc_id in enumerate(docs):
            logger.info(f"Adding document {doc_id} to queue with priority {i}")
            queue_id = db_manager.enqueue_document(document_id=doc_id, priority=i)
            queue_ids.append(queue_id)

        # Verify queue stats
        stats = db_manager.get_queue_stats()
        logger.info(f"Queue stats after adding documents: {stats}")

        self.assertEqual(stats['pending'], len(docs),
                         f"Expected {len(docs)} pending documents, got {stats['pending']}")

        # Simulate application restart by creating a new database manager
        logger.info("Simulating application restart...")
        time.sleep(1)  # Brief pause

        # Create a new database manager (simulates application restart)
        new_db_manager = DatabaseManager()

        # Check queue stats after "restart"
        stats = new_db_manager.get_queue_stats()
        logger.info(f"Queue stats after 'restart': {stats}")

        self.assertEqual(stats['pending'], len(docs),
                         f"Queue didn't persist through restart. Expected {len(docs)} pending documents, got {stats['pending']}")

        # Dequeue a document
        logger.info("Dequeuing a document after restart...")
        dequeued = new_db_manager.dequeue_document(count=1)

        self.assertTrue(dequeued, "Failed to dequeue document after restart")
        self.assertEqual(len(dequeued), 1, "Expected 1 dequeued document")

        queue_item_id, doc_id = dequeued[0]
        logger.info(f"Dequeued document {doc_id} with queue item ID {queue_item_id}")

        # Verify document status changed to "processing"
        doc = new_db_manager.get_document(doc_id)
        self.assertEqual(doc['status'], "processing",
                         f"Document status not updated correctly. Expected 'processing', got '{doc['status']}'")
        logger.info("Document status correctly updated to 'processing'")

        # Complete the queue item
        logger.info("Completing the queue item...")
        success = new_db_manager.complete_queue_item(queue_item_id, success=True)

        self.assertTrue(success, "Failed to complete queue item")

        # Update document status to completed
        new_db_manager.update_document_status(doc_id, "completed")

        # Verify queue stats again
        stats = new_db_manager.get_queue_stats()
        logger.info(f"Final queue stats: {stats}")

        self.assertEqual(stats['completed'], 1, "Expected 1 completed queue item")
        self.assertEqual(stats['processing'], 0, "Expected 0 processing queue items")

        logger.info("Processing queue persistence test passed")

    def test_checkpoint_functionality(self):
        """Test checkpoint save/load functionality."""
        logger.info("=== Testing Checkpoint Functionality ===")

        # Create a checkpoint manager
        checkpoint_manager = CheckpointManager()

        # Create a test process ID
        process_id = f"test_process_{uuid.uuid4().hex[:8]}"

        # Create a test state
        state = {
            "process_type": "test",
            "document_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "processed_ids": ["doc1", "doc2"],
            "current_position": 2,
            "total_count": 5,
            "custom_data": {
                "test_key": "test_value",
                "nested": {
                    "value": 123
                }
            }
        }

        # Save the checkpoint
        logger.info(f"Saving checkpoint for process {process_id}")
        saved = checkpoint_manager.save_checkpoint(process_id, state)

        self.assertTrue(saved, "Failed to save checkpoint")

        # Simulate restart
        logger.info("Simulating application restart...")
        time.sleep(1)

        # Create a new checkpoint manager (simulates application restart)
        new_checkpoint_manager = CheckpointManager()

        # Load the checkpoint
        logger.info(f"Loading checkpoint for process {process_id}")
        loaded_state = new_checkpoint_manager.load_checkpoint(process_id)

        self.assertIsNotNone(loaded_state, "Failed to load checkpoint")

        # Verify checkpoint data
        self.assertEqual(loaded_state.get('document_ids'), state['document_ids'],
                         "Loaded document_ids don't match saved data")
        self.assertEqual(loaded_state.get('processed_ids'), state['processed_ids'],
                         "Loaded processed_ids don't match saved data")
        self.assertEqual(loaded_state.get('current_position'), state['current_position'],
                         "Loaded current_position doesn't match saved data")

        logger.info("Checkpoint functionality test passed")

        # Clean up test checkpoint
        deleted = new_checkpoint_manager.delete_checkpoint(process_id)
        self.assertTrue(deleted, "Failed to delete checkpoint")
        logger.info(f"Cleaned up test checkpoint for process {process_id}")


if __name__ == "__main__":
    unittest.main()