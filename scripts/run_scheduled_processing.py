#!/usr/bin/env python
"""
Script to run scheduled processing of the embedding pipeline.
This script can be called by cron jobs or task scheduler.
"""
import os
import sys
import logging
import datetime
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from src after path modification
from src.config import load_config
from src.cli.commands import main_async
import asyncio


def setup_logging(log_dir, log_level="INFO"):
    """Set up logging for the scheduled task."""
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create log file name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"scheduled_run_{timestamp}.log")

    # Configure logging
    level = getattr(logging, log_level)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run scheduled pipeline processing')

    parser.add_argument('--mode', choices=['auto', 'bulk', 'incremental'],
                        default='auto', help='Processing mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')

    return parser.parse_args()


async def run_scheduled_processing(args):
    """Run the scheduled processing with the specified args."""
    # Create mock args for the CLI
    class MockArgs:
        def __init__(self):
            self.command = 'process'
            self.mode = args.mode
            self.workers = None
            self.verbose = False
            self.file = None
            self.directory = None
            self.resume = False
            self.batch_size = None
            self.adaptive_scaling = True
            self.force = False

    # Load configuration
    config = load_config()

    # Initialize CLI
    from src.cli.commands import EmbeddingPipelineCLI
    cli = EmbeddingPipelineCLI(config)

    # Run process command
    mock_args = MockArgs()
    return await cli.process_command(mock_args)


async def main():
    """Main entry point for the scheduled processing script."""
    # Parse arguments
    args = parse_args()

    # Set up project path
    project_path = Path(__file__).parent.parent
    log_dir = project_path / "logs" / "scheduled"

    # Set up logging
    log_file = setup_logging(log_dir, args.log_level)

    # Log start
    logging.info("Starting scheduled pipeline run")
    logging.info(f"Mode: {args.mode}")

    try:
        # Set appropriate event loop policy for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Run the processing
        success = await run_scheduled_processing(args)

        if success:
            logging.info("Scheduled pipeline run completed successfully")
            return 0
        else:
            logging.error("Scheduled pipeline run failed")
            return 1

    except Exception as e:
        logging.exception(f"Unhandled exception in scheduled run: {str(e)}")
        return 1
    finally:
        logging.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)