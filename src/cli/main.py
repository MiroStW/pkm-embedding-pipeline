#!/usr/bin/env python
"""
Main entry point for the embedding pipeline CLI.
"""
import asyncio
import logging
import sys
import os

from src.config import load_config
from src.cli.commands import main_async

def setup_logging(config):
    """Set up logging based on configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('log_file')
    console_output = log_config.get('console_output', True)

    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logging
    handlers = []
    if console_output:
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def main():
    """Main entry point for the CLI."""
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config)

    logging.debug("Starting CLI with configuration loaded")

    # Set up appropriate event loop policy for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the async CLI
    exit_code = asyncio.run(main_async(config))

    # Exit with appropriate code
    sys.exit(exit_code)

if __name__ == "__main__":
    main()