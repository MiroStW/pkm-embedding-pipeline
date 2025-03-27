"""
Worker pool implementation for the pipeline orchestration.
Handles adaptive scaling of worker processes based on system load and queue size.
"""
import os
import asyncio
import logging
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Dict

class WorkerPool:
    """
    Manages a pool of worker threads for document processing.
    Dynamically scales the number of workers based on system load.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the worker pool with configuration settings.

        Args:
            config: Configuration dictionary with worker settings
        """
        self.config = config
        self.min_workers = config.get('pipeline', {}).get('min_workers', 2)
        self.max_workers = config.get('pipeline', {}).get('max_workers', os.cpu_count())
        self.target_cpu_usage = config.get('pipeline', {}).get('target_cpu_usage', 70)
        self.scaling_interval = config.get('pipeline', {}).get('scaling_interval', 10)
        self.executor = ThreadPoolExecutor(max_workers=self.min_workers)
        self.current_workers = self.min_workers
        self.logger = logging.getLogger(__name__)
        self.active_tasks = 0
        self.adaptive_scaling = config.get('pipeline', {}).get('adaptive_scaling', True)
        self.monitor_task = None

    async def start_monitoring(self):
        """Start the resource monitoring task for adaptive scaling."""
        if self.adaptive_scaling and self.monitor_task is None:
            self.monitor_task = asyncio.create_task(self._monitor_resources())
            self.logger.info(f"Started worker pool with {self.current_workers} workers (adaptive scaling enabled)")
        else:
            self.logger.info(f"Started worker pool with {self.current_workers} workers (fixed size)")

    async def _monitor_resources(self):
        """Monitor system resources and adjust worker count accordingly."""
        while True:
            await asyncio.sleep(self.scaling_interval)

            # Get current CPU usage
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            queue_size = self.active_tasks

            # Calculate ideal worker count based on load
            if cpu_usage < self.target_cpu_usage - 20 and queue_size > self.current_workers:
                # Increase workers if CPU usage is low and we have queued tasks
                new_count = min(self.current_workers + 1, self.max_workers)
                if new_count > self.current_workers:
                    self._scale_workers(new_count)
            elif cpu_usage > self.target_cpu_usage + 10:
                # Decrease workers if CPU usage is high
                new_count = max(self.current_workers - 1, self.min_workers)
                if new_count < self.current_workers:
                    self._scale_workers(new_count)

            self.logger.debug(f"Resource monitor: CPU: {cpu_usage}%, Memory: {memory_usage}%, "
                             f"Active tasks: {self.active_tasks}, Workers: {self.current_workers}")

    def _scale_workers(self, new_count: int):
        """
        Scale the worker pool to the specified size.

        Args:
            new_count: New number of workers
        """
        old_count = self.current_workers
        self.current_workers = new_count

        # Create a new executor with the updated worker count
        new_executor = ThreadPoolExecutor(max_workers=new_count)
        old_executor = self.executor
        self.executor = new_executor

        # Shutdown the old executor gracefully (after tasks complete)
        old_executor.shutdown(wait=False)

        self.logger.info(f"Scaled worker pool from {old_count} to {new_count} workers")

    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Submit a task to the worker pool.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Result of the function execution
        """
        self.active_tasks += 1
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                lambda: func(*args, **kwargs)
            )
            return result
        finally:
            self.active_tasks -= 1

    async def shutdown(self, wait: bool = True):
        """
        Shut down the worker pool.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.executor.shutdown(wait=wait)
        self.logger.info("Worker pool shut down")

    @property
    def capacity(self) -> int:
        """Get the current worker capacity."""
        return self.current_workers