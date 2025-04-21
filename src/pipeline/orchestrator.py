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
        # NOTE: This is a coroutine, so it needs to be awaited before use
        self.vector_db_coro = create_vector_db_uploader(config)
        self.vector_db = None  # Will be initialized in run()

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
                    print(f"DEBUG: Consumer {consumer_id} processing file: {file_path}")
                    try:
                        # Process document
                        result = await self.worker_pool.submit(
                            self.document_processor.process_file,
                            file_path
                        )
                        print(f"DEBUG: Consumer {consumer_id} got result for {file_path}: {result}")

                        # Update tracking database
                        if result and result.get('status') == 'success':
                            print(f"DEBUG: Consumer {consumer_id} proceeding to index {file_path}")
                            # Generate embeddings and store in vector database
                            try:
                                # Ensure vector_db is available (not None)
                                if self.vector_db is None:
                                    self.logger.error(f"Vector database not initialized")
                                    raise RuntimeError("Vector database not initialized")

                                # Index the document DIRECTLY (not via worker pool)
                                embedding_result = await self.vector_db.index_document(result)

                                # Check if indexing was successful (optional, based on index_document return)
                                if not embedding_result:
                                    self.logger.error(f"Failed to index document {file_path}")
                                    self.document_db.mark_error(file_path, "Failed during indexing step")
                                    self.error_count += 1
                                    self.error_files.add(file_path)
                                    continue # Skip to next file in batch

                                # Update document status if indexing successful
                                self.document_db.mark_completed(file_path, result['metadata'])

                                # Update tracking
                                self.processed_count += 1
                                self.processed_files.add(file_path)

                                # Log success
                                self.logger.info(f"Successfully processed and indexed: {file_path}")


                            except Exception as e:
                                # Handle vector database errors
                                self.logger.error(f"Error indexing file {file_path}: {str(e)}")
                                self.document_db.mark_error(file_path, f"Indexing error: {str(e)}")
                                self.error_count += 1
                                self.error_files.add(file_path)
                        else:
                            print(f"DEBUG: Consumer {consumer_id} NOT indexing {file_path} - Status was not 'success' or result empty.")
                            # Mark as error
                            error_msg = result.get('error', 'Unknown error or non-success status') if result else 'Empty result from process_file'
                            self.document_db.mark_error(file_path, error_msg)
                            self.error_count += 1
                            self.error_files.add(file_path)

                    except Exception as e:
                        # Handle general processing errors
                        self.logger.error(f"Exception processing file {file_path}: {str(e)}")
                        self.document_db.mark_error(file_path, str(e))
                        self.error_count += 1
                        self.error_files.add(file_path)

                # Mark batch as done
                self.queue.task_done()

            except Exception as e:
                self.logger.error(f"Unhandled exception in consumer {consumer_id}: {str(e)}")
                # Mark batch as done even if there was an error
                self.queue.task_done()

        self.logger.info(f"Consumer {consumer_id} completed")

    async def run(self, files_to_process: List[str]) -> Dict[str, Any]:
        """
        Run the pipeline with the specified files.

        Args:
            files_to_process: List of file paths to process

        Returns:
            Dictionary with results of the pipeline run
        """
        self.start_time = time.time()
        self.running = True
        self.processed_count = 0
        self.error_count = 0
        self.processed_files = set()
        self.error_files = set()

        # Initialize the vector database uploader
        try:
            self.vector_db = await self.vector_db_coro
            if self.vector_db is None:
                self.logger.error("Failed to initialize vector database uploader")
                return {
                    "status": "error",
                    "error": "Failed to initialize vector database uploader"
                }
        except Exception as e:
            self.logger.error(f"Error initializing vector database uploader: {str(e)}")
            return {
                "status": "error",
                "error": f"Error initializing vector database uploader: {str(e)}"
            }

        # Start consumers
        for i in range(self.consumer_count):
            consumer = asyncio.create_task(self.consumer_task(i))
            self.consumers.append(consumer)

        # Start producer
        self.producer = asyncio.create_task(self.producer_task(files_to_process))

        # Wait for completion
        await self.producer
        await asyncio.gather(*self.consumers)

        # Clean up
        self.running = False
        self.end_time = time.time()
        self.worker_pool.shutdown()

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint("pipeline_global", {
            "last_run": time.time(),
            "processed_files": len(self.processed_files),
            "error_files": len(self.error_files),
            "total_files": len(files_to_process)
        })

        # Return results
        return {
            "status": "completed",
            "elapsed_time": self.end_time - self.start_time,
            "total_files": len(files_to_process),
            "processed_files": len(self.processed_files),
            "error_files": len(self.error_files),
            "throughput": len(self.processed_files) / (self.end_time - self.start_time) if self.end_time > self.start_time else 0
        }

    async def resume_from_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Resume pipeline execution from the last checkpoint.

        Returns:
            Results of the pipeline run, or None if no checkpoint is available
        """
        checkpoint = self.checkpoint_manager.load_checkpoint("pipeline_global")
        if not checkpoint:
            self.logger.warning("No checkpoint found, cannot resume")
            return None

        # Get pending and error files
        pending_files = self.document_db.get_pending_documents()
        error_files = self.document_db.get_error_documents()

        files_to_process = pending_files + error_files
        if not files_to_process:
            self.logger.warning("No pending or error files to process")
            return None

        self.logger.info(f"Resuming from checkpoint with {len(files_to_process)} files")
        return await self.run(files_to_process)