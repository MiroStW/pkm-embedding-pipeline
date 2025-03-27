"""
Resilience system for the embedding pipeline.

This module integrates error handling, logging, monitoring, and recovery
to provide a comprehensive resilience system for the embedding pipeline.
"""

import asyncio
import functools
import inspect
import os
import signal
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

from ..pipeline import error_handling, logging, monitoring, recovery


# Type definitions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ResilienceManager:
    """Manager for integrating error handling, logging, monitoring, and recovery."""

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        metrics_dir: Union[str, Path] = "logs/metrics",
        recovery_dir: Union[str, Path] = "data/recovery",
        log_level: int = logging.logging.INFO,
        checkpoint_interval: int = 60,  # seconds
        max_checkpoints: int = 5,
        autosave_metrics_interval: int = 60,  # seconds
    ):
        """
        Initialize the resilience manager.

        Args:
            log_dir: Directory to store log files
            metrics_dir: Directory to store metrics files
            recovery_dir: Directory to store recovery files
            log_level: Logging level
            checkpoint_interval: Interval in seconds for automatic checkpointing
            max_checkpoints: Maximum number of checkpoints to keep
            autosave_metrics_interval: Interval in seconds for automatic saving of metrics
        """
        # Initialize core components
        self.logger_manager = logging.init_logging(
            log_dir=log_dir,
            log_level=log_level,
        )
        self.monitor = monitoring.init_monitoring(
            metrics_dir=metrics_dir,
            autosave_interval=autosave_metrics_interval,
        )
        self.recovery_manager = recovery.init_recovery(
            recovery_dir=recovery_dir,
            checkpoint_interval=checkpoint_interval,
            max_checkpoints=max_checkpoints,
        )

        # Get a logger for this manager
        self.logger = logging.get_logger("pipeline.resilience")

        # Start background processes
        self.monitor.start_autosave()

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        # Save original handlers
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        # Register our handlers
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals by saving state before exiting."""
        self.logger.warning(f"Received signal {signum}, performing graceful shutdown")

        try:
            # Save any active metrics
            self.monitor.cleanup()

            # Let the user know we're shutting down
            print("\nGracefully shutting down, saving state...")

            # Restore original signal handlers
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)

            # Re-raise the signal to let the default handler take over
            signal.raise_signal(signum)
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
            sys.exit(1)

    def cleanup(self) -> None:
        """Clean up resources used by the resilience manager."""
        self.monitor.cleanup()

    @contextmanager
    def resilient_run(
        self,
        run_name: str,
        run_id: Optional[str] = None,
        total_documents: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        recover_from_failure: bool = True,
    ):
        """
        Context manager for resilient pipeline runs.

        This creates a dedicated logger and metrics for the run,
        sets up error handling and recovery, and ensures proper cleanup.

        Args:
            run_name: Name of the run
            run_id: Unique identifier for the run (generated if not provided)
            total_documents: Total number of documents to process
            metadata: Additional metadata for the run
            recover_from_failure: Whether to attempt recovery from previous failures

        Yields:
            Dictionary with context for the run:
                - run_id: Unique identifier for the run
                - logger: Logger for the run
                - recovery_manager: Recovery manager instance
                - monitor: Monitor instance
                - checkpoint: Current checkpoint (if any)
        """
        run_id = run_id or f"{run_name}-{uuid.uuid4().hex[:8]}"
        run_logger = logging.create_run_log(run_name)

        try:
            # Check for recovery state if requested
            checkpoint = None
            if recover_from_failure:
                recovery_state, checkpoint = self.recovery_manager.get_recovery_state(run_id)
                if recovery_state == recovery.RecoveryState.PARTIAL:
                    run_logger.info(f"Recovering from previous partial run: {run_id}")
                    # If we have a checkpoint, use its document count
                    if checkpoint:
                        total_documents = checkpoint.total_documents

            # Start monitoring
            metrics = self.monitor.start_run(run_id, total_documents)

            # Create a new checkpoint if we don't have one
            if checkpoint is None:
                checkpoint = self.recovery_manager.create_checkpoint(
                    run_id=run_id,
                    metadata=metadata or {},
                )

            # Yield context for the run
            run_logger.info(f"Starting resilient run: {run_id}")
            yield {
                "run_id": run_id,
                "logger": run_logger,
                "recovery_manager": self.recovery_manager,
                "monitor": self.monitor,
                "checkpoint": checkpoint,
            }

            # Mark the run as complete
            self.monitor.complete_run(run_id)
            run_logger.info(f"Completed resilient run: {run_id}")

        except Exception as e:
            run_logger.error(f"Error in resilient run {run_id}: {e}")
            run_logger.error(traceback.format_exc())
            self.monitor.fail_run(run_id)
            # Re-raise the exception for the caller to handle
            raise

    def resilient_function(
        self,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        log_errors: bool = True,
        failure_threshold: Optional[int] = None,
    ) -> Callable[[F], F]:
        """
        Decorator for making functions resilient.

        This combines error handling, retries, and logging.

        Args:
            retry_count: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            log_errors: Whether to log errors
            failure_threshold: Maximum number of failures before aborting (None = no limit)

        Returns:
            Decorated function
        """
        def decorator(func: F) -> F:
            # Track function failures
            failures = {'count': 0}

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    # Use retry decorator
                    retried_func = error_handling.retry(
                        max_retries=retry_count,
                        retry_delay=retry_delay,
                    )(func)

                    result = await retried_func(*args, **kwargs)
                    failures['count'] = 0  # Reset failure count on success
                    return result

                except Exception as e:
                    failures['count'] += 1

                    if log_errors:
                        logger = logging.get_logger(func.__module__)
                        logger.error(f"Error in {func.__name__}: {e}")
                        logger.debug(traceback.format_exc())

                    # Check failure threshold
                    if failure_threshold is not None and failures['count'] >= failure_threshold:
                        raise error_handling.ProcessingError(
                            message=f"Failure threshold exceeded for {func.__name__}: {failures['count']} failures",
                            severity=error_handling.ErrorSeverity.CRITICAL,
                            original_exception=e,
                        )

                    raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    # Use retry decorator
                    retried_func = error_handling.retry(
                        max_retries=retry_count,
                        retry_delay=retry_delay,
                    )(func)

                    result = retried_func(*args, **kwargs)
                    failures['count'] = 0  # Reset failure count on success
                    return result

                except Exception as e:
                    failures['count'] += 1

                    if log_errors:
                        logger = logging.get_logger(func.__module__)
                        logger.error(f"Error in {func.__name__}: {e}")
                        logger.debug(traceback.format_exc())

                    # Check failure threshold
                    if failure_threshold is not None and failures['count'] >= failure_threshold:
                        raise error_handling.ProcessingError(
                            message=f"Failure threshold exceeded for {func.__name__}: {failures['count']} failures",
                            severity=error_handling.ErrorSeverity.CRITICAL,
                            original_exception=e,
                        )

                    raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def resilient_task(
        self,
        stage: monitoring.PipelineStage,
        run_id: str,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> Callable[[F], F]:
        """
        Decorator for making pipeline tasks resilient.

        This integrates with monitoring and error handling.

        Args:
            stage: Pipeline stage this task belongs to
            run_id: Identifier for the current run
            retry_count: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds

        Returns:
            Decorated function
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get item being processed (if any)
                item_id = None
                if args and hasattr(args[0], 'id'):
                    item_id = args[0].id
                elif 'item' in kwargs and hasattr(kwargs['item'], 'id'):
                    item_id = kwargs['item'].id

                try:
                    # Update stage metrics
                    self.monitor.update_stage(
                        run_id=run_id,
                        stage=stage,
                        items_processed=0,
                        items_pending=1,
                    )

                    # Use retry decorator
                    retried_func = error_handling.retry(
                        max_retries=retry_count,
                        retry_delay=retry_delay,
                    )(func)

                    # Execute the function
                    result = await retried_func(*args, **kwargs)

                    # Update stage metrics
                    self.monitor.update_stage(
                        run_id=run_id,
                        stage=stage,
                        items_processed=1,
                        items_pending=0,
                    )

                    return result

                except Exception as e:
                    # Update stage metrics
                    error_msg = f"Error processing {item_id or 'item'}: {e}"
                    self.monitor.update_stage(
                        run_id=run_id,
                        stage=stage,
                        items_failed=1,
                        items_pending=0,
                        error=error_msg,
                    )

                    # Log the error
                    logger = logging.get_logger(func.__module__)
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())

                    raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get item being processed (if any)
                item_id = None
                if args and hasattr(args[0], 'id'):
                    item_id = args[0].id
                elif 'item' in kwargs and hasattr(kwargs['item'], 'id'):
                    item_id = kwargs['item'].id

                try:
                    # Update stage metrics
                    self.monitor.update_stage(
                        run_id=run_id,
                        stage=stage,
                        items_processed=0,
                        items_pending=1,
                    )

                    # Use retry decorator
                    retried_func = error_handling.retry(
                        max_retries=retry_count,
                        retry_delay=retry_delay,
                    )(func)

                    # Execute the function
                    result = retried_func(*args, **kwargs)

                    # Update stage metrics
                    self.monitor.update_stage(
                        run_id=run_id,
                        stage=stage,
                        items_processed=1,
                        items_pending=0,
                    )

                    return result

                except Exception as e:
                    # Update stage metrics
                    error_msg = f"Error processing {item_id or 'item'}: {e}"
                    self.monitor.update_stage(
                        run_id=run_id,
                        stage=stage,
                        items_failed=1,
                        items_pending=0,
                        error=error_msg,
                    )

                    # Log the error
                    logger = logging.get_logger(func.__module__)
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())

                    raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator


# Singleton instance
_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager() -> ResilienceManager:
    """
    Get the global resilience manager.

    Returns:
        Global resilience manager instance
    """
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


def init_resilience(
    log_dir: Union[str, Path] = "logs",
    metrics_dir: Union[str, Path] = "logs/metrics",
    recovery_dir: Union[str, Path] = "data/recovery",
    log_level: int = logging.logging.INFO,
    checkpoint_interval: int = 60,
    max_checkpoints: int = 5,
    autosave_metrics_interval: int = 60,
) -> ResilienceManager:
    """
    Initialize the global resilience manager.

    Args:
        log_dir: Directory to store log files
        metrics_dir: Directory to store metrics files
        recovery_dir: Directory to store recovery files
        log_level: Logging level
        checkpoint_interval: Interval in seconds for automatic checkpointing
        max_checkpoints: Maximum number of checkpoints to keep
        autosave_metrics_interval: Interval in seconds for automatic saving of metrics

    Returns:
        Global resilience manager instance
    """
    global _resilience_manager
    _resilience_manager = ResilienceManager(
        log_dir=log_dir,
        metrics_dir=metrics_dir,
        recovery_dir=recovery_dir,
        log_level=log_level,
        checkpoint_interval=checkpoint_interval,
        max_checkpoints=max_checkpoints,
        autosave_metrics_interval=autosave_metrics_interval,
    )
    return _resilience_manager


# Convenience function for getting a resilient run context
def resilient_run(
    run_name: str,
    run_id: Optional[str] = None,
    total_documents: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    recover_from_failure: bool = True,
):
    """
    Context manager for resilient pipeline runs.

    See ResilienceManager.resilient_run for details.
    """
    return get_resilience_manager().resilient_run(
        run_name=run_name,
        run_id=run_id,
        total_documents=total_documents,
        metadata=metadata,
        recover_from_failure=recover_from_failure,
    )


# Convenience decorators
def resilient_function(
    retry_count: int = 3,
    retry_delay: float = 1.0,
    log_errors: bool = True,
    failure_threshold: Optional[int] = None,
) -> Callable[[F], F]:
    """
    Decorator for making functions resilient.

    See ResilienceManager.resilient_function for details.
    """
    return get_resilience_manager().resilient_function(
        retry_count=retry_count,
        retry_delay=retry_delay,
        log_errors=log_errors,
        failure_threshold=failure_threshold,
    )


def resilient_task(
    stage: monitoring.PipelineStage,
    run_id: str,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> Callable[[F], F]:
    """
    Decorator for making pipeline tasks resilient.

    See ResilienceManager.resilient_task for details.
    """
    return get_resilience_manager().resilient_task(
        stage=stage,
        run_id=run_id,
        retry_count=retry_count,
        retry_delay=retry_delay,
    )