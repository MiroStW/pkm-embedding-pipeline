"""
Logging system for the embedding pipeline.

This module provides a comprehensive logging system for the embedding pipeline,
including log configuration, log rotation, and log formatting.
"""

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# Default date format for logs
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class PipelineLogger:
    """Logger manager for the embedding pipeline."""

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        log_level: int = logging.INFO,
        log_format: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: bool = True,
        log_file_prefix: str = "embedding-pipeline",
    ):
        """
        Initialize the logger manager.

        Args:
            log_dir: Directory to store log files
            log_level: Logging level
            log_format: Format string for log messages
            date_format: Format string for dates in log messages
            max_bytes: Maximum size of each log file before rotation
            backup_count: Number of backup log files to keep
            console_output: Whether to output logs to console
            log_file_prefix: Prefix for log files
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.log_format = log_format
        self.date_format = date_format
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.console_output = console_output
        self.log_file_prefix = log_file_prefix

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        self._configure_root_logger()

        # Keep track of created loggers
        self._loggers: Dict[str, logging.Logger] = {}

    def _configure_root_logger(self) -> None:
        """Configure the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add handlers
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(
                logging.Formatter(self.log_format, self.date_format)
            )
            root_logger.addHandler(console_handler)

        # Main log file
        main_log_file = self.log_dir / f"{self.log_file_prefix}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(
            logging.Formatter(self.log_format, self.date_format)
        )
        root_logger.addHandler(file_handler)

        # Error log file (for ERROR and CRITICAL)
        error_log_file = self.log_dir / f"{self.log_file_prefix}-error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(self.log_format, self.date_format)
        )
        root_logger.addHandler(error_handler)

    def get_logger(
        self,
        name: str,
        level: Optional[int] = None,
        add_timestamp_to_name: bool = False,
    ) -> logging.Logger:
        """
        Get a logger with the given name.

        Args:
            name: Name of the logger
            level: Logging level (defaults to the level set for the manager)
            add_timestamp_to_name: Whether to add a timestamp to the logger name
                                  (useful for distinguishing multiple runs)

        Returns:
            Logger instance
        """
        if add_timestamp_to_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            full_name = f"{name}-{timestamp}"
        else:
            full_name = name

        if full_name in self._loggers:
            return self._loggers[full_name]

        logger = logging.getLogger(full_name)

        if level is not None:
            logger.setLevel(level)

        self._loggers[full_name] = logger
        return logger

    def create_run_log(
        self,
        run_name: str,
        include_timestamp: bool = True,
    ) -> logging.Logger:
        """
        Create a logger for a specific pipeline run.

        This creates a dedicated log file for this run in addition to the main log.

        Args:
            run_name: Name of the run
            include_timestamp: Whether to include a timestamp in the log file name

        Returns:
            Logger for this run
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") if include_timestamp else ""
        log_name = f"{run_name}-{timestamp}" if include_timestamp else run_name

        # Create a new logger
        logger = logging.getLogger(log_name)
        logger.setLevel(self.log_level)

        # Create a dedicated log file for this run
        run_log_file = self.log_dir / f"{self.log_file_prefix}-{log_name}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            run_log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(
            logging.Formatter(self.log_format, self.date_format)
        )
        logger.addHandler(file_handler)

        self._loggers[log_name] = logger
        return logger


# Singleton instance
_logger_manager: Optional[PipelineLogger] = None


def init_logging(
    log_dir: Union[str, Path] = "logs",
    log_level: int = logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_output: bool = True,
    log_file_prefix: str = "embedding-pipeline",
) -> PipelineLogger:
    """
    Initialize the global logger manager.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        log_format: Format string for log messages
        date_format: Format string for dates in log messages
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output logs to console
        log_file_prefix: Prefix for log files

    Returns:
        The global logger manager
    """
    global _logger_manager
    _logger_manager = PipelineLogger(
        log_dir=log_dir,
        log_level=log_level,
        log_format=log_format,
        date_format=date_format,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console_output=console_output,
        log_file_prefix=log_file_prefix,
    )
    return _logger_manager


def get_logger(
    name: str,
    level: Optional[int] = None,
    add_timestamp_to_name: bool = False,
) -> logging.Logger:
    """
    Get a logger with the given name from the global manager.

    Args:
        name: Name of the logger
        level: Logging level (defaults to the level set for the manager)
        add_timestamp_to_name: Whether to add a timestamp to the logger name

    Returns:
        Logger instance
    """
    global _logger_manager
    if _logger_manager is None:
        init_logging()
    return _logger_manager.get_logger(name, level, add_timestamp_to_name)


def create_run_log(
    run_name: str,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Create a logger for a specific pipeline run.

    Args:
        run_name: Name of the run
        include_timestamp: Whether to include a timestamp in the log file name

    Returns:
        Logger for this run
    """
    global _logger_manager
    if _logger_manager is None:
        init_logging()
    return _logger_manager.create_run_log(run_name, include_timestamp)