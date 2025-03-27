"""
Recovery system for the embedding pipeline.

This module provides functionality for recovering from pipeline failures,
including checkpoint creation and restoration, state management,
and automatic recovery procedures.
"""

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..pipeline import logging


class RecoveryState(str, Enum):
    """Enum representing different recovery states."""
    CLEAN = "clean"                  # No recovery needed
    PARTIAL = "partial"              # Partial recovery needed
    FULL = "full"                    # Full recovery needed
    CORRUPTED = "corrupted"          # Recovery state is corrupted


@dataclass
class DocumentState:
    """State information for a document being processed."""
    doc_id: str
    filepath: str
    chunks_processed: bool = False
    embeddings_generated: bool = False
    db_uploaded: bool = False
    error: Optional[str] = None
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_chunks_processed(self) -> None:
        """Mark this document as having its chunks processed."""
        self.chunks_processed = True
        self.last_updated = time.time()

    def mark_embeddings_generated(self) -> None:
        """Mark this document as having its embeddings generated."""
        self.embeddings_generated = True
        self.last_updated = time.time()

    def mark_db_uploaded(self) -> None:
        """Mark this document as having been uploaded to the database."""
        self.db_uploaded = True
        self.last_updated = time.time()

    def set_error(self, error: str) -> None:
        """Set an error for this document."""
        self.error = error
        self.last_updated = time.time()

    def clear_error(self) -> None:
        """Clear any error for this document."""
        self.error = None
        self.last_updated = time.time()

    def reset(self) -> None:
        """Reset the processing state for this document."""
        self.chunks_processed = False
        self.embeddings_generated = False
        self.db_uploaded = False
        self.error = None
        self.last_updated = time.time()

    def is_completed(self) -> bool:
        """Check if this document has completed processing."""
        return self.db_uploaded

    def is_in_progress(self) -> bool:
        """Check if this document is partially processed."""
        return (self.chunks_processed or self.embeddings_generated) and not self.db_uploaded

    def has_error(self) -> bool:
        """Check if this document has an error."""
        return self.error is not None


@dataclass
class PipelineCheckpoint:
    """Checkpoint for the pipeline state."""
    checkpoint_id: str
    timestamp: float = field(default_factory=time.time)
    document_states: Dict[str, DocumentState] = field(default_factory=dict)
    run_id: Optional[str] = None
    completed_documents: Set[str] = field(default_factory=set)
    failed_documents: Set[str] = field(default_factory=set)
    pending_documents: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_document(self, doc_state: DocumentState) -> None:
        """Add a document state to this checkpoint."""
        self.document_states[doc_state.doc_id] = doc_state

        if doc_state.is_completed():
            self.completed_documents.add(doc_state.doc_id)
            self.pending_documents.discard(doc_state.doc_id)
            self.failed_documents.discard(doc_state.doc_id)
        elif doc_state.has_error():
            self.failed_documents.add(doc_state.doc_id)
            self.pending_documents.discard(doc_state.doc_id)
            self.completed_documents.discard(doc_state.doc_id)
        else:
            self.pending_documents.add(doc_state.doc_id)
            self.completed_documents.discard(doc_state.doc_id)
            self.failed_documents.discard(doc_state.doc_id)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document state from this checkpoint."""
        if doc_id in self.document_states:
            del self.document_states[doc_id]
            self.completed_documents.discard(doc_id)
            self.failed_documents.discard(doc_id)
            self.pending_documents.discard(doc_id)

    def update_document(self, doc_state: DocumentState) -> None:
        """Update a document state in this checkpoint."""
        self.add_document(doc_state)

    def get_document_state(self, doc_id: str) -> Optional[DocumentState]:
        """Get the state of a document from this checkpoint."""
        return self.document_states.get(doc_id)

    @property
    def total_documents(self) -> int:
        """Get the total number of documents in this checkpoint."""
        return len(self.document_states)

    @property
    def progress(self) -> float:
        """Get the progress percentage for this checkpoint."""
        if self.total_documents == 0:
            return 0.0
        return (len(self.completed_documents) / self.total_documents) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert this checkpoint to a dictionary."""
        result = {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "total_documents": self.total_documents,
            "completed_documents": len(self.completed_documents),
            "failed_documents": len(self.failed_documents),
            "pending_documents": len(self.pending_documents),
            "progress": self.progress,
            "metadata": self.metadata,
        }
        return result


class RecoveryManager:
    """Manager for pipeline recovery and checkpointing."""

    def __init__(
        self,
        recovery_dir: Union[str, Path] = "data/recovery",
        checkpoint_interval: int = 60,  # seconds
        max_checkpoints: int = 5,
    ):
        """
        Initialize the recovery manager.

        Args:
            recovery_dir: Directory to store recovery files
            checkpoint_interval: Interval in seconds for automatic checkpointing
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.recovery_dir = Path(recovery_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.logger = logging.get_logger("pipeline.recovery")

        # Create recovery directory if it doesn't exist
        self.recovery_dir.mkdir(parents=True, exist_ok=True)

        # Current run checkpoint
        self.current_checkpoint: Optional[PipelineCheckpoint] = None
        self.last_checkpoint_time: float = 0

    def create_checkpoint(
        self,
        run_id: str,
        document_states: Optional[Dict[str, DocumentState]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineCheckpoint:
        """
        Create a new checkpoint.

        Args:
            run_id: Identifier for the run
            document_states: Dictionary of document states to include
            metadata: Additional metadata for the checkpoint

        Returns:
            The new checkpoint
        """
        checkpoint_id = f"{run_id}-{int(time.time())}"
        checkpoint = PipelineCheckpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            metadata=metadata or {},
        )

        if document_states:
            for doc_state in document_states.values():
                checkpoint.add_document(doc_state)

        self.current_checkpoint = checkpoint
        self.last_checkpoint_time = time.time()

        # Save the checkpoint
        self._save_checkpoint(checkpoint)

        self.logger.info(f"Created checkpoint: {checkpoint_id}, documents: {checkpoint.total_documents}")
        return checkpoint

    def update_checkpoint(
        self,
        document_state: DocumentState,
        force_save: bool = False,
    ) -> None:
        """
        Update the current checkpoint with a document state.

        Args:
            document_state: Document state to update
            force_save: Whether to force saving the checkpoint regardless of interval
        """
        if self.current_checkpoint is None:
            self.logger.warning("No current checkpoint to update")
            return

        self.current_checkpoint.update_document(document_state)

        # Check if we should save the checkpoint
        current_time = time.time()
        if force_save or (current_time - self.last_checkpoint_time) >= self.checkpoint_interval:
            self._save_checkpoint(self.current_checkpoint)
            self.last_checkpoint_time = current_time

    def _save_checkpoint(self, checkpoint: PipelineCheckpoint) -> None:
        """
        Save a checkpoint to disk.

        Args:
            checkpoint: Checkpoint to save
        """
        checkpoint_path = self.recovery_dir / f"{checkpoint.checkpoint_id}.pickle"

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)

            # Also save a JSON summary
            summary_path = self.recovery_dir / f"{checkpoint.checkpoint_id}.json"
            with open(summary_path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

            self._cleanup_old_checkpoints()
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints, keeping only the most recent ones."""
        if self.current_checkpoint is None or self.current_checkpoint.run_id is None:
            return

        run_id = self.current_checkpoint.run_id
        checkpoint_files = sorted(
            [f for f in self.recovery_dir.glob(f"{run_id}-*.pickle")],
            key=lambda f: os.path.getmtime(f),
            reverse=True,
        )

        if len(checkpoint_files) > self.max_checkpoints:
            for file_to_remove in checkpoint_files[self.max_checkpoints:]:
                try:
                    file_to_remove.unlink()
                    # Also remove the JSON summary if it exists
                    json_file = file_to_remove.with_suffix(".json")
                    if json_file.exists():
                        json_file.unlink()
                except Exception as e:
                    self.logger.error(f"Error removing old checkpoint {file_to_remove}: {e}")

    def load_checkpoint(self, checkpoint_id: str) -> Optional[PipelineCheckpoint]:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_id: Identifier for the checkpoint

        Returns:
            The loaded checkpoint, or None if not found or corrupted
        """
        checkpoint_path = self.recovery_dir / f"{checkpoint_id}.pickle"

        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return None

        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)

            self.current_checkpoint = checkpoint
            self.last_checkpoint_time = time.time()

            self.logger.info(f"Loaded checkpoint: {checkpoint_id}, documents: {checkpoint.total_documents}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            return None

    def load_latest_checkpoint(self, run_id: Optional[str] = None) -> Optional[PipelineCheckpoint]:
        """
        Load the latest checkpoint for a run.

        Args:
            run_id: Identifier for the run, or None to load the latest checkpoint for any run

        Returns:
            The latest checkpoint, or None if not found
        """
        pattern = f"{run_id}-*.pickle" if run_id else "*.pickle"
        checkpoint_files = sorted(
            [f for f in self.recovery_dir.glob(pattern)],
            key=lambda f: os.path.getmtime(f),
            reverse=True,
        )

        if not checkpoint_files:
            self.logger.warning(f"No checkpoints found{f' for run {run_id}' if run_id else ''}")
            return None

        latest_checkpoint_id = checkpoint_files[0].stem
        return self.load_checkpoint(latest_checkpoint_id)

    def get_recovery_state(self, run_id: Optional[str] = None) -> Tuple[RecoveryState, Optional[PipelineCheckpoint]]:
        """
        Get the recovery state for a run.

        Args:
            run_id: Identifier for the run, or None to check for any run

        Returns:
            Tuple of recovery state and the latest checkpoint (if any)
        """
        checkpoint = self.load_latest_checkpoint(run_id)

        if checkpoint is None:
            return RecoveryState.CLEAN, None

        if checkpoint.failed_documents:
            return RecoveryState.PARTIAL, checkpoint

        if checkpoint.pending_documents:
            return RecoveryState.PARTIAL, checkpoint

        return RecoveryState.CLEAN, checkpoint

    def clear_recovery_data(self, run_id: Optional[str] = None) -> None:
        """
        Clear recovery data for a run.

        Args:
            run_id: Identifier for the run, or None to clear all recovery data
        """
        pattern = f"{run_id}-*.pickle" if run_id else "*.pickle"
        for checkpoint_file in self.recovery_dir.glob(pattern):
            try:
                checkpoint_file.unlink()
                # Also remove the JSON summary if it exists
                json_file = checkpoint_file.with_suffix(".json")
                if json_file.exists():
                    json_file.unlink()
            except Exception as e:
                self.logger.error(f"Error removing checkpoint {checkpoint_file}: {e}")

        self.current_checkpoint = None
        self.last_checkpoint_time = 0

        self.logger.info(f"Cleared recovery data{f' for run {run_id}' if run_id else ''}")

    def list_checkpoints(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            run_id: Identifier for the run, or None to list all checkpoints

        Returns:
            List of checkpoint summaries
        """
        pattern = f"{run_id}-*.json" if run_id else "*.json"
        checkpoint_files = sorted(
            [f for f in self.recovery_dir.glob(pattern)],
            key=lambda f: os.path.getmtime(f),
            reverse=True,
        )

        result = []
        for checkpoint_file in checkpoint_files:
            try:
                with open(checkpoint_file, "r") as f:
                    checkpoint_summary = json.load(f)
                    result.append(checkpoint_summary)
            except Exception as e:
                self.logger.error(f"Error reading checkpoint summary {checkpoint_file}: {e}")

        return result


# Singleton instance
_recovery_manager: Optional[RecoveryManager] = None


def get_recovery_manager() -> RecoveryManager:
    """
    Get the global recovery manager.

    Returns:
        Global recovery manager instance
    """
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager()
    return _recovery_manager


def init_recovery(
    recovery_dir: Union[str, Path] = "data/recovery",
    checkpoint_interval: int = 60,
    max_checkpoints: int = 5,
) -> RecoveryManager:
    """
    Initialize the global recovery manager.

    Args:
        recovery_dir: Directory to store recovery files
        checkpoint_interval: Interval in seconds for automatic checkpointing
        max_checkpoints: Maximum number of checkpoints to keep

    Returns:
        Global recovery manager instance
    """
    global _recovery_manager
    _recovery_manager = RecoveryManager(
        recovery_dir=recovery_dir,
        checkpoint_interval=checkpoint_interval,
        max_checkpoints=max_checkpoints,
    )
    return _recovery_manager