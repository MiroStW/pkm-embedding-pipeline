"""
Error handling and resilience module for the embedding pipeline.

This module provides comprehensive error handling, retry mechanisms,
and failure recovery for the embedding pipeline.
"""

import asyncio
import functools
import logging
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

# Type definitions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Enum to classify error severity levels."""
    INFO = "INFO"               # Non-critical, informational
    WARNING = "WARNING"         # Concerning but non-fatal
    ERROR = "ERROR"             # Operation failed but pipeline can continue
    CRITICAL = "CRITICAL"       # Fatal error requiring intervention


class ProcessingError(Exception):
    """Base exception class for pipeline processing errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        original_exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.original_exception = original_exception
        self.context = context or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        error_info = f"{self.severity.value}: {self.message}"
        if self.original_exception:
            error_info += f"\nCaused by: {type(self.original_exception).__name__}: {str(self.original_exception)}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            error_info += f"\nContext: {context_str}"
        return error_info


class DocumentProcessingError(ProcessingError):
    """Error during document parsing or chunking."""
    pass


class EmbeddingGenerationError(ProcessingError):
    """Error during embedding generation."""
    pass


class DatabaseError(ProcessingError):
    """Error with database operations."""
    pass


class VectorDBError(ProcessingError):
    """Error with vector database (Pinecone) operations."""
    pass


class GitIntegrationError(ProcessingError):
    """Error with git integration."""
    pass


class RetryableError(ProcessingError):
    """Error that can be retried."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        original_exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(message, severity, original_exception, context)
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def should_retry(self) -> bool:
        """Determine if another retry should be attempted."""
        return self.retry_count < self.max_retries


def retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_exceptions: List[Type[Exception]] = None,
):
    """
    Decorator for retrying a function or method on specific exceptions.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for increasing delay between retries
        retry_exceptions: List of exception types that should trigger a retry
                         (defaults to RetryableError)
    """
    retry_exceptions = retry_exceptions or [RetryableError]

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            retries = 0
            current_delay = retry_delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except tuple(retry_exceptions) as e:
                    retries += 1

                    if isinstance(e, RetryableError):
                        e.retry_count = retries
                        should_retry = e.should_retry()
                    else:
                        should_retry = retries < max_retries

                    if not should_retry:
                        raise

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after error: {str(e)}. "
                        f"Retrying in {current_delay:.2f}s"
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            retries = 0
            current_delay = retry_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except tuple(retry_exceptions) as e:
                    retries += 1

                    if isinstance(e, RetryableError):
                        e.retry_count = retries
                        should_retry = e.should_retry()
                    else:
                        should_retry = retries < max_retries

                    if not should_retry:
                        raise

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after error: {str(e)}. "
                        f"Retrying in {current_delay:.2f}s"
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def safe_execute(
    error_type: Type[ProcessingError] = ProcessingError,
    default_value: Any = None,
    log_error: bool = True,
    raise_error: bool = False,
) -> Callable[[F], F]:
    """
    Decorator for safely executing functions and handling exceptions.

    Args:
        error_type: The error type to use when wrapping exceptions
        default_value: Value to return on failure if not raising exception
        log_error: Whether to log the error
        raise_error: Whether to raise the error after logging
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error executing {func.__name__}: {str(e)}"
                context = {"args": args, "kwargs": kwargs}

                wrapped_error = error_type(
                    message=error_msg,
                    original_exception=e,
                    context=context
                )

                if log_error:
                    logger = logging.getLogger(__name__)
                    logger.error(str(wrapped_error))
                    logger.debug(traceback.format_exc())

                if raise_error:
                    raise wrapped_error

                return default_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error executing {func.__name__}: {str(e)}"
                context = {"args": args, "kwargs": kwargs}

                wrapped_error = error_type(
                    message=error_msg,
                    original_exception=e,
                    context=context
                )

                if log_error:
                    logger = logging.getLogger(__name__)
                    logger.error(str(wrapped_error))
                    logger.debug(traceback.format_exc())

                if raise_error:
                    raise wrapped_error

                return default_value

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator