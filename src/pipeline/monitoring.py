"""
Monitoring system for the embedding pipeline.

This module provides monitoring capabilities for the embedding pipeline,
including metrics collection, performance tracking, and status reporting.
"""

import asyncio
import json
import time
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..pipeline import logging


class PipelineStage(str, Enum):
    """Enum representing different stages of the pipeline."""
    DOCUMENT_PROCESSING = "document_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    DATABASE_UPDATE = "database_update"
    GIT_INTEGRATION = "git_integration"
    PIPELINE_ORCHESTRATION = "pipeline_orchestration"


class PipelineStatus(str, Enum):
    """Enum representing the status of a pipeline run."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageMetrics:
    """Metrics for a specific pipeline stage."""
    stage: PipelineStage
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    items_processed: int = 0
    items_failed: int = 0
    items_pending: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get the duration of this stage in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def is_complete(self) -> bool:
        """Check if this stage is complete."""
        return self.end_time is not None

    @property
    def success_rate(self) -> float:
        """Get the success rate of this stage."""
        total = self.items_processed + self.items_failed
        return (self.items_processed / total) * 100 if total > 0 else 0

    def complete(self) -> None:
        """Mark this stage as complete."""
        self.end_time = time.time()

    def add_error(self, error: str) -> None:
        """Add an error message to this stage."""
        self.errors.append(error)
        self.items_failed += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert this stage to a dictionary."""
        result = asdict(self)
        # Add computed properties
        result["duration"] = self.duration
        result["is_complete"] = self.is_complete
        result["success_rate"] = self.success_rate
        return result


@dataclass
class PipelineMetrics:
    """Metrics for a pipeline run."""
    run_id: str
    status: PipelineStatus = PipelineStatus.INITIALIZED
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    stage_metrics: Dict[PipelineStage, StageMetrics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize after construction."""
        for stage in PipelineStage:
            if stage not in self.stage_metrics:
                self.stage_metrics[stage] = StageMetrics(stage=stage)

    @property
    def duration(self) -> float:
        """Get the duration of this run in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def overall_progress(self) -> float:
        """Get the overall progress of this run as a percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100

    @property
    def current_stage(self) -> Optional[PipelineStage]:
        """Get the current active stage of the pipeline."""
        for stage in PipelineStage:
            metrics = self.stage_metrics[stage]
            if not metrics.is_complete and metrics.items_processed > 0:
                return stage
        return None

    def start(self) -> None:
        """Start tracking this pipeline run."""
        self.status = PipelineStatus.RUNNING
        self.start_time = time.time()

    def complete(self) -> None:
        """Mark this pipeline run as complete."""
        self.status = PipelineStatus.COMPLETED
        self.end_time = time.time()
        # Complete any active stages
        for metrics in self.stage_metrics.values():
            if not metrics.is_complete and metrics.items_processed > 0:
                metrics.complete()

    def fail(self) -> None:
        """Mark this pipeline run as failed."""
        self.status = PipelineStatus.FAILED
        self.end_time = time.time()

    def pause(self) -> None:
        """Pause this pipeline run."""
        self.status = PipelineStatus.PAUSED

    def resume(self) -> None:
        """Resume this pipeline run."""
        self.status = PipelineStatus.RUNNING

    def update_documents(
        self,
        total: Optional[int] = None,
        processed: Optional[int] = None,
        failed: Optional[int] = None
    ) -> None:
        """Update document counts."""
        if total is not None:
            self.total_documents = total
        if processed is not None:
            self.processed_documents = processed
        if failed is not None:
            self.failed_documents = failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert this run to a dictionary."""
        result = {
            "run_id": self.run_id,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "failed_documents": self.failed_documents,
            "overall_progress": self.overall_progress,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "stages": {stage.value: metrics.to_dict() for stage, metrics in self.stage_metrics.items()},
        }
        return result


class PipelineMonitor:
    """Monitor for tracking pipeline execution metrics and status."""

    def __init__(
        self,
        metrics_dir: Union[str, Path] = "logs/metrics",
        autosave_interval: int = 60,  # seconds
    ):
        """
        Initialize the pipeline monitor.

        Args:
            metrics_dir: Directory to store metrics files
            autosave_interval: Interval in seconds for automatic saving of metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.autosave_interval = autosave_interval
        self.active_runs: Dict[str, PipelineMetrics] = {}
        self.completed_runs: Set[str] = set()
        self.logger = logging.get_logger("pipeline.monitor")

        # Create metrics directory if it doesn't exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Background task for autosaving metrics
        self._autosave_task = None
        self._autosave_thread = None
        self._stop_event = threading.Event()

    async def _autosave_loop(self) -> None:
        """Background task for periodically saving metrics using asyncio."""
        while True:
            try:
                for run_id, metrics in list(self.active_runs.items()):
                    self.save_metrics(run_id)
                await asyncio.sleep(self.autosave_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in autosave loop: {e}")
                await asyncio.sleep(self.autosave_interval)

    def _autosave_thread_func(self) -> None:
        """Background thread function for periodically saving metrics."""
        while not self._stop_event.is_set():
            try:
                for run_id, metrics in list(self.active_runs.items()):
                    self.save_metrics(run_id)
                time.sleep(self.autosave_interval)
            except Exception as e:
                self.logger.error(f"Error in autosave thread: {e}")
                time.sleep(self.autosave_interval)

    def start_autosave(self) -> None:
        """Start the autosave background task."""
        # Use a thread instead of asyncio task to avoid event loop issues
        if self._autosave_thread is None:
            self._stop_event.clear()
            self._autosave_thread = threading.Thread(
                target=self._autosave_thread_func,
                daemon=True,
            )
            self._autosave_thread.start()

    def stop_autosave(self) -> None:
        """Stop the autosave background task."""
        if self._autosave_thread:
            self._stop_event.set()
            self._autosave_thread.join(timeout=1.0)
            self._autosave_thread = None

        if self._autosave_task:
            self._autosave_task.cancel()
            self._autosave_task = None

    def start_run(self, run_id: str, total_documents: int = 0) -> PipelineMetrics:
        """
        Start tracking a new pipeline run.

        Args:
            run_id: Unique identifier for this run
            total_documents: Total number of documents to process

        Returns:
            Metrics object for the new run
        """
        metrics = PipelineMetrics(run_id=run_id, total_documents=total_documents)
        metrics.start()
        self.active_runs[run_id] = metrics
        self.logger.info(f"Started pipeline run: {run_id}")
        return metrics

    def get_metrics(self, run_id: str) -> Optional[PipelineMetrics]:
        """
        Get metrics for a pipeline run.

        Args:
            run_id: Identifier for the run

        Returns:
            Metrics object for the run, or None if not found
        """
        return self.active_runs.get(run_id)

    def complete_run(self, run_id: str) -> None:
        """
        Mark a pipeline run as complete.

        Args:
            run_id: Identifier for the run
        """
        if run_id in self.active_runs:
            metrics = self.active_runs[run_id]
            metrics.complete()
            self.save_metrics(run_id)
            self.completed_runs.add(run_id)
            self.logger.info(f"Completed pipeline run: {run_id}, duration: {metrics.duration:.2f}s")

    def fail_run(self, run_id: str) -> None:
        """
        Mark a pipeline run as failed.

        Args:
            run_id: Identifier for the run
        """
        if run_id in self.active_runs:
            metrics = self.active_runs[run_id]
            metrics.fail()
            self.save_metrics(run_id)
            self.completed_runs.add(run_id)
            self.logger.error(f"Failed pipeline run: {run_id}, duration: {metrics.duration:.2f}s")

    def update_stage(
        self,
        run_id: str,
        stage: PipelineStage,
        items_processed: Optional[int] = None,
        items_failed: Optional[int] = None,
        items_pending: Optional[int] = None,
        error: Optional[str] = None,
        complete: bool = False,
    ) -> None:
        """
        Update metrics for a pipeline stage.

        Args:
            run_id: Identifier for the run
            stage: Pipeline stage to update
            items_processed: Number of items processed
            items_failed: Number of items failed
            items_pending: Number of items pending
            error: Error message if any
            complete: Whether to mark the stage as complete
        """
        if run_id not in self.active_runs:
            return

        metrics = self.active_runs[run_id]
        stage_metrics = metrics.stage_metrics[stage]

        if items_processed is not None:
            stage_metrics.items_processed = items_processed

        if items_failed is not None:
            stage_metrics.items_failed = items_failed

        if items_pending is not None:
            stage_metrics.items_pending = items_pending

        if error:
            stage_metrics.add_error(error)

        if complete:
            stage_metrics.complete()

    def generate_report(self, run_id: str) -> Dict[str, Any]:
        """
        Generate a detailed report for a pipeline run.

        Args:
            run_id: Identifier for the run

        Returns:
            Dictionary containing detailed metrics and status
        """
        if run_id not in self.active_runs:
            return {}

        metrics = self.active_runs[run_id]
        return metrics.to_dict()

    def save_metrics(self, run_id: str) -> None:
        """
        Save metrics for a pipeline run to a file.

        Args:
            run_id: Identifier for the run
        """
        if run_id not in self.active_runs:
            return

        metrics = self.active_runs[run_id]
        timestamp = datetime.fromtimestamp(metrics.start_time).strftime("%Y%m%d-%H%M%S")
        filename = self.metrics_dir / f"{run_id}-{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

    def load_metrics(self, filename: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load metrics from a file.

        Args:
            filename: Path to the metrics file

        Returns:
            Dictionary containing metrics, or None if the file doesn't exist
        """
        filepath = Path(filename)
        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def cleanup(self) -> None:
        """Cleanup resources used by the monitor."""
        self.stop_autosave()

        # Save all active metrics
        for run_id in list(self.active_runs.keys()):
            self.save_metrics(run_id)


# Singleton instance
_monitor: Optional[PipelineMonitor] = None


def get_monitor() -> PipelineMonitor:
    """
    Get the global pipeline monitor.

    Returns:
        Global pipeline monitor instance
    """
    global _monitor
    if _monitor is None:
        _monitor = PipelineMonitor()
    return _monitor


def init_monitoring(
    metrics_dir: Union[str, Path] = "logs/metrics",
    autosave_interval: int = 60,
) -> PipelineMonitor:
    """
    Initialize the global pipeline monitor.

    Args:
        metrics_dir: Directory to store metrics files
        autosave_interval: Interval in seconds for automatic saving of metrics

    Returns:
        Global pipeline monitor instance
    """
    global _monitor
    _monitor = PipelineMonitor(
        metrics_dir=metrics_dir,
        autosave_interval=autosave_interval,
    )
    return _monitor