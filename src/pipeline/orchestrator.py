"""
Pipeline orchestrator implementing producer-consumer architecture with
bulk, incremental, and auto-detection processing modes.
"""
import asyncio
import logging
import time
from enum import Enum
from typing import List, Dict, Any, Optional

from src.database.document_db import DocumentTracker
from src.database.vector_db_factory import create_vector_db_uploader
from src.database.checkpoint import CheckpointManager
from src.processors import DocumentProcessor
from .worker_pool import WorkerPool

class ProcessingMode(Enum):
    """Enumeration of supported processing modes."""
    BULK = "bulk"
    INCREMENTAL = "incremental"
    AUTO = "auto"

class PipelineOrchestrator:
    """
    Orchestrates the document processing pipeline with producer-consumer architecture.
    Supports bulk, incremental, and auto-detection processing modes.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.document_db = DocumentTracker(config.get('database', {}).get('tracking_db_path'))

        # Use factory to create vector DB uploader
        self.vector_db = create_vector_db_uploader(config)
        if self.vector_db is None:
            self.logger.warning("Vector database is not configured correctly, using mock implementation")
            from src.database.mock_vector_db import MockVectorDatabaseUploader
            self.vector_db = MockVectorDatabaseUploader()

        self.checkpoint_manager = CheckpointManager(config.get('database', {}).get('checkpoint_dir', 'data/checkpoints'))
        self.document_processor = DocumentProcessor(config)
        self.worker_pool = WorkerPool(config)

        # Queue for work items
        self.queue = asyncio.Queue()

        # Set processing mode
        mode_str = config.get('pipeline', {}).get('processing_mode', 'auto')
        self.processing_mode = ProcessingMode(mode_str)

        # Pipeline state
        self.running = False
        self.consumers = []
        self.producer = None
        self.processed_count = 0
        self.error_count = 0
        self.processed_files = set()
        self.error_files = set()

        # Performance tracking
        self.start_time = None
        self.end_time = None

        # Configure pipeline parameters
        self.batch_size = config.get('pipeline', {}).get('batch_size', 10)
        self.max_queue_size = config.get('pipeline', {}).get('max_queue_size', 100)
        self.consumer_count = config.get('pipeline', {}).get('consumer_count', 2)

    def detect_processing_mode(self, files_to_process: List[str]) -> ProcessingMode:
        """
        Auto-detect the optimal processing mode based on the number of files.

        Args:
            files_to_process: List of file paths to process

        Returns:
            The detected processing mode
        """
        if self.processing_mode != ProcessingMode.AUTO:
            return self.processing_mode

        # Get total number of documents
        total_files = len(files_to_process)

        # Detect mode based on number of files
        if total_files > 100:
            self.logger.info(f"Auto-detected BULK mode for {total_files} files")
            return ProcessingMode.BULK
        else:
            self.logger.info(f"Auto-detected INCREMENTAL mode for {total_files} files")
            return ProcessingMode.INCREMENTAL

    async def producer_task(self, files_to_process: List[str]):
        """
        Producer task that feeds files into the processing queue.

        Args:
            files_to_process: List of file paths to process
        """
        self.logger.info(f"Producer started with {len(files_to_process)} files")

        # Process files in batches for bulk mode or individually for incremental
        active_mode = self.detect_processing_mode(files_to_process)

        if active_mode == ProcessingMode.BULK:
            # Bulk mode: Process in batches to maximize throughput
            batch = []
            for file_path in files_to_process:
                batch.append(file_path)

                if len(batch) >= self.batch_size:
                    # Wait if queue is getting too full
                    while self.queue.qsize() >= self.max_queue_size:
                        await asyncio.sleep(0.5)

                    await self.queue.put(batch.copy())
                    batch.clear()

                    # Record progress
                    progress = (self.processed_count / len(files_to_process)) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({self.processed_count}/{len(files_to_process)})")

            # Put any remaining files
            if batch:
                await self.queue.put(batch)

        else:  # INCREMENTAL mode
            # Incremental mode: Process files individually with careful tracking
            for file_path in files_to_process:
                # Check if file has already been processed
                if self.document_db.is_processed(file_path):
                    self.logger.debug(f"Skipping already processed file: {file_path}")
                    continue

                await self.queue.put([file_path])

        # Signal end of work
        for _ in range(self.consumer_count):
            await self.queue.put(None)

        self.logger.info("Producer task completed")

    async def consumer_task(self, consumer_id: int):
        """
        Consumer task that processes files from the queue.

        Args:
            consumer_id: ID of the consumer task
        """
        self.logger.info(f"Consumer {consumer_id} started")

        while True:
            # Get next batch of files
            batch = await self.queue.get()

            # Check for end signal
            if batch is None:
                self.logger.info(f"Consumer {consumer_id} received end signal")
                break

            try:
                # Process each file in the batch
                for file_path in batch:
                    try:
                        # Process document
                        result = await self.worker_pool.submit(
                            self.document_processor.process_file,
                            file_path
                        )

                        # Update tracking database
                        if result['status'] == 'success':
                            # Generate embeddings and store in vector database
                            embedding_result = await self.worker_pool.submit(
                                self.vector_db.index_document,
                                result
                            )

                            # Update document status
                            self.document_db.mark_completed(file_path, result['metadata'])

                            # Update tracking
                            self.processed_count += 1
                            self.processed_files.add(file_path)

                            # Log success
                            self.logger.info(f"Successfully processed: {file_path}")
                        else:
                            # Mark as error
                            self.document_db.mark_error(file_path, result.get('error', 'Unknown error'))
                            self.error_count += 1
                            self.error_files.add(file_path)
                            self.logger.error(f"Error processing file {file_path}: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        # Handle unexpected errors
                        self.error_count += 1
                        self.error_files.add(file_path)
                        self.logger.exception(f"Exception processing file {file_path}: {str(e)}")

                        # Update database
                        self.document_db.mark_error(file_path, str(e))

                # Create checkpoint after each batch
                self.checkpoint_manager.save_checkpoint(
                    processed_files=self.processed_files,
                    error_files=self.error_files,
                    processed_count=self.processed_count,
                    error_count=self.error_count
                )

            finally:
                # Mark batch as done
                self.queue.task_done()

        self.logger.info(f"Consumer {consumer_id} completed")

    async def run(self, files_to_process: List[str]) -> Dict[str, Any]:
        """
        Run the pipeline on the specified files.

        Args:
            files_to_process: List of file paths to process

        Returns:
            Dictionary with pipeline execution statistics
        """
        if self.running:
            raise RuntimeError("Pipeline is already running")

        self.running = True
        self.start_time = time.time()

        try:
            # Initialize worker pool
            await self.worker_pool.start_monitoring()

            # Start consumers
            self.consumers = []
            for i in range(self.consumer_count):
                consumer = asyncio.create_task(self.consumer_task(i))
                self.consumers.append(consumer)

            # Start producer
            self.producer = asyncio.create_task(self.producer_task(files_to_process))

            # Wait for producer to complete
            await self.producer

            # Wait for all consumers to complete
            await asyncio.gather(*self.consumers)

            # Shutdown worker pool
            await self.worker_pool.shutdown()

            # Calculate statistics
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            throughput = self.processed_count / elapsed_time if elapsed_time > 0 else 0

            return {
                "status": "completed",
                "total_files": len(files_to_process),
                "processed_files": self.processed_count,
                "error_files": self.error_count,
                "elapsed_time": elapsed_time,
                "throughput": throughput
            }

        except Exception as e:
            self.logger.exception(f"Pipeline execution failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "processed_files": self.processed_count,
                "error_files": self.error_count
            }
        finally:
            self.running = False

    async def resume_from_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Resume pipeline execution from the last checkpoint.

        Returns:
            Dictionary with pipeline execution statistics or None if no checkpoint
        """
        # Load checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if not checkpoint:
            self.logger.info("No checkpoint found to resume from")
            return None

        # Extract checkpoint data
        processed_files = set(checkpoint.get('processed_files', []))
        error_files = set(checkpoint.get('error_files', []))

        # Get all files that need processing
        all_files = self.document_db.get_all_files()

        # Filter out already processed files
        files_to_process = [f for f in all_files if f not in processed_files and f not in error_files]

        self.logger.info(f"Resuming from checkpoint: {len(processed_files)} processed, "
                        f"{len(error_files)} errors, {len(files_to_process)} remaining")

        # Initialize state from checkpoint
        self.processed_files = processed_files
        self.error_files = error_files
        self.processed_count = len(processed_files)
        self.error_count = len(error_files)

        # Run pipeline with remaining files
        if files_to_process:
            return await self.run(files_to_process)
        else:
            self.logger.info("No files remaining to process")
            total_files = len(processed_files) + len(error_files)
            return {
                "status": "completed",
                "total_files": total_files,
                "processed_files": self.processed_count,
                "error_files": self.error_count,
                "elapsed_time": 0,
                "throughput": 0
            }