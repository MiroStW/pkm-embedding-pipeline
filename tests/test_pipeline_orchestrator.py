"""
Unit tests for the pipeline orchestrator implementation.
"""
import os
import sys
import unittest
import asyncio
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from src.pipeline import PipelineOrchestrator
from src.database.document_db import DocumentTracker
from src.database.mock_vector_db import MockVectorDatabaseUploader
from src.database.checkpoint import CheckpointManager

class TestPipelineOrchestrator(unittest.TestCase):
    """Test cases for the pipeline orchestrator implementation."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Create test configuration
        self.config = {
            'pipeline': {
                'min_workers': 1,
                'max_workers': 2,
                'adaptive_scaling': False,
                'processing_mode': 'auto',
                'batch_size': 2,
                'max_queue_size': 10,
                'consumer_count': 1
            },
            'database': {
                'tracking_db_path': os.path.join(self.temp_dir, 'test_tracking.db'),
                'vector_db': {
                    'provider': 'memory',
                    'api_key': 'test_key',
                    'environment': 'test_env',
                    'index_name': 'test_index'
                }
            },
            'embedding': {
                'model_type': 'distiluse',
                'device': 'cpu',
                'dimension': 768
            },
            'logging': {
                'level': 'DEBUG',
                'console_output': True
            }
        }

        # Create test files
        self.test_files = []
        for i in range(5):
            file_path = os.path.join(self.temp_dir, f"test_file_{i}.md")
            with open(file_path, 'w') as f:
                f.write(f"---\ntitle: Test File {i}\nid: test_{i}\n---\n\n# Test Content {i}\n\nThis is test file {i}.")
            self.test_files.append(file_path)

        # Set up asyncio event loop
        if sys.platform == 'win32':
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
            self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Initialize mocks
        self.mock_document_db = MagicMock(spec=DocumentTracker)
        self.mock_vector_db = MagicMock(spec=MockVectorDatabaseUploader)
        self.mock_checkpoint_manager = MagicMock(spec=CheckpointManager)

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

        # Close event loop
        self.loop.close()

    @patch('src.pipeline.orchestrator.DocumentTracker')
    @patch('src.pipeline.orchestrator.create_vector_db_uploader')
    @patch('src.pipeline.orchestrator.CheckpointManager')
    @patch('src.pipeline.orchestrator.DocumentProcessor')
    @patch('src.pipeline.orchestrator.WorkerPool')
    def test_orchestrator_init(self, mock_worker_pool, mock_processor,
                              mock_checkpoint, mock_vector_db_factory, mock_document_db):
        """Test pipeline orchestrator initialization."""
        # Arrange
        mock_worker_pool.return_value = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_checkpoint.return_value = self.mock_checkpoint_manager
        mock_vector_db_factory.return_value = self.mock_vector_db
        mock_document_db.return_value = self.mock_document_db

        # Act
        orchestrator = PipelineOrchestrator(self.config)

        # Assert
        self.assertEqual(orchestrator.batch_size, 2)
        self.assertEqual(orchestrator.consumer_count, 1)
        mock_document_db.assert_called_once_with(self.config.get('database', {}).get('tracking_db_path'))

        # Check vector_db_factory was called with correct config
        mock_vector_db_factory.assert_called_once_with(self.config.get('database', {}))

        mock_checkpoint.assert_called_once_with(self.config.get('database', {}).get('checkpoint_dir', 'data/checkpoints'))
        mock_processor.assert_called_once_with(self.config)
        mock_worker_pool.assert_called_once_with(self.config)

    @patch('src.pipeline.orchestrator.DocumentTracker')
    @patch('src.pipeline.orchestrator.VectorDatabaseUploader')
    @patch('src.pipeline.orchestrator.CheckpointManager')
    @patch('src.pipeline.orchestrator.DocumentProcessor')
    @patch('src.pipeline.orchestrator.WorkerPool')
    def test_processing_mode_detection(self, mock_worker_pool, mock_processor,
                                      mock_checkpoint, mock_vector_db, mock_document_db):
        """Test processing mode auto-detection."""
        # Arrange
        mock_worker_pool.return_value = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_checkpoint.return_value = self.mock_checkpoint_manager
        mock_vector_db.return_value = self.mock_vector_db
        mock_document_db.return_value = self.mock_document_db

        orchestrator = PipelineOrchestrator(self.config)

        # Create small file list (should be incremental)
        small_list = [f"file_{i}.md" for i in range(5)]

        # Create large file list (should be bulk)
        large_list = [f"file_{i}.md" for i in range(200)]

        # Act
        small_mode = orchestrator.detect_processing_mode(small_list)
        large_mode = orchestrator.detect_processing_mode(large_list)

        # Assert
        self.assertEqual(small_mode.value, "incremental")
        self.assertEqual(large_mode.value, "bulk")

    @patch('src.pipeline.orchestrator.DocumentTracker')
    @patch('src.pipeline.orchestrator.VectorDatabaseUploader')
    @patch('src.pipeline.orchestrator.CheckpointManager')
    @patch('src.pipeline.orchestrator.DocumentProcessor')
    @patch('src.pipeline.orchestrator.WorkerPool')
    def test_run_pipeline(self, mock_worker_pool, mock_processor,
                         mock_checkpoint, mock_vector_db, mock_document_db):
        """Test running the pipeline."""
        # Arrange
        mock_worker_pool_instance = MagicMock()
        mock_worker_pool.return_value = mock_worker_pool_instance

        # Configure the async methods in the mock
        async def mock_start_monitoring():
            return None

        async def mock_submit(func, *args, **kwargs):
            return {'status': 'success', 'metadata': {'id': 'test_1'}, 'chunks': []}

        async def mock_shutdown(wait=True):
            return None

        mock_worker_pool_instance.start_monitoring.side_effect = mock_start_monitoring
        mock_worker_pool_instance.submit.side_effect = mock_submit
        mock_worker_pool_instance.shutdown.side_effect = mock_shutdown

        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.process_file.return_value = {
            'status': 'success',
            'metadata': {'id': 'test_1', 'title': 'Test File'},
            'chunks': [{'content': 'Test content', 'metadata': {}}]
        }

        mock_checkpoint.return_value = self.mock_checkpoint_manager
        self.mock_checkpoint_manager.save_checkpoint.return_value = True

        mock_vector_db.return_value = self.mock_vector_db
        self.mock_vector_db.index_document.return_value = True

        mock_document_db.return_value = self.mock_document_db
        mock_document_db.is_processed.return_value = False
        mock_document_db.mark_completed.return_value = True

        orchestrator = PipelineOrchestrator(self.config)

        # Prepare to run the pipeline
        async def run_test():
            return await orchestrator.run(self.test_files[:2])

        # Act
        result = self.loop.run_until_complete(run_test())

        # Assert
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['total_files'], 2)
        self.assertGreater(result['elapsed_time'], 0)

    @patch('src.pipeline.orchestrator.DocumentTracker')
    @patch('src.pipeline.orchestrator.VectorDatabaseUploader')
    @patch('src.pipeline.orchestrator.CheckpointManager')
    @patch('src.pipeline.orchestrator.DocumentProcessor')
    @patch('src.pipeline.orchestrator.WorkerPool')
    def test_resume_from_checkpoint(self, mock_worker_pool, mock_processor,
                                   mock_checkpoint, mock_vector_db, mock_document_db):
        """Test resuming pipeline from checkpoint."""
        # Arrange
        # Configure the mocks
        mock_worker_pool_instance = MagicMock()
        mock_worker_pool.return_value = mock_worker_pool_instance

        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance

        mock_checkpoint.return_value = self.mock_checkpoint_manager

        # Only one processed file (the first test file)
        processed_files = [self.test_files[0]]

        self.mock_checkpoint_manager.load_checkpoint.return_value = {
            'processed_files': processed_files,
            'error_files': [],
            'processed_count': 1,
            'error_count': 0
        }

        mock_vector_db.return_value = self.mock_vector_db
        mock_document_db.return_value = self.mock_document_db
        mock_document_db.get_all_files.return_value = self.test_files

        orchestrator = PipelineOrchestrator(self.config)

        # For this test, we'll skip the actual run method
        # and verify the proper initialization of state from the checkpoint

        # Prepare to run the test
        async def run_test():
            result = await orchestrator.resume_from_checkpoint()
            print(f"Resume result: {result}")
            return result

        # Act
        result = self.loop.run_until_complete(run_test())

        # Assert
        # The pipeline should have initialized its state from the checkpoint
        self.assertEqual(orchestrator.processed_count, 1)
        self.assertEqual(len(orchestrator.processed_files), 1)
        self.assertEqual(len(orchestrator.error_files), 0)

        # The total_files in the result should be consistent with the checkpoint
        # In this case, 1 processed file and 0 error files, so total is 1
        self.assertEqual(result['total_files'], 1)
        self.assertEqual(result['processed_files'], 1)
        self.assertEqual(result['error_files'], 0)
        self.assertEqual(result['status'], 'completed')

if __name__ == '__main__':
    unittest.main()