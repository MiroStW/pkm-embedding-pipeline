"""
Pipeline module for the embedding pipeline.

This module provides the core functionality for the embedding pipeline,
including document processing, embedding generation, and vector database integration.
"""

# Import the main components
from . import error_handling
from . import logging
from . import monitoring
from . import recovery
from . import resilience

# Export commonly used functions and classes for easier access
from .error_handling import retry, safe_execute, ProcessingError, ErrorSeverity
from .logging import get_logger, create_run_log, init_logging
from .monitoring import PipelineStage, PipelineStatus, get_monitor, init_monitoring
from .recovery import RecoveryState, DocumentState, get_recovery_manager, init_recovery
from .resilience import (
    resilient_run,
    resilient_function,
    resilient_task,
    get_resilience_manager,
    init_resilience,
)

from src.pipeline.worker_pool import WorkerPool
from src.pipeline.orchestrator import PipelineOrchestrator

__all__ = ['PipelineOrchestrator', 'WorkerPool']