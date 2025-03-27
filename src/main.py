"""
Main entry point for the PKM Chatbot embedding pipeline.
"""
import os
import sys
import argparse
import logging
import yaml
import json
import asyncio
from src.database.init_db import init_db
from src.processors import DocumentProcessor
from src.pipeline import PipelineOrchestrator

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

    logging.info("Logging initialized")

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        # Load base config
        config = yaml.safe_load(file)

        # Override with environment variables if they exist
        for section in config:
            if isinstance(config[section], dict):
                for key, value in config[section].items():
                    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                        env_var = value[2:-1]
                        if env_var in os.environ:
                            config[section][key] = os.environ[env_var]

    return config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='PKM Chatbot Embedding Pipeline')
    parser.add_argument('--mode', choices=['bulk', 'incremental', 'auto'], default='auto',
                        help='Processing mode: bulk, incremental, or auto')
    parser.add_argument('--workers', type=int, help='Number of worker processes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--test', action='store_true', help='Run document processor test')
    parser.add_argument('--file', help='Process a single file')
    parser.add_argument('--directory', help='Process a directory')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--batch-size', type=int, help='Batch size for bulk processing')
    parser.add_argument('--adaptive-scaling', action='store_true', help='Enable adaptive worker pool scaling')

    return parser.parse_args()

async def run_pipeline(config, args):
    """Run the pipeline with the specified configuration."""
    logging.info("Initializing pipeline orchestrator")
    orchestrator = PipelineOrchestrator(config)

    # Determine files to process
    files_to_process = []

    if args.file:
        # Process a single file
        if os.path.exists(args.file):
            files_to_process = [args.file]
        else:
            logging.error(f"File not found: {args.file}")
            return False
    elif args.directory:
        # Process all markdown files in a directory (recursively)
        if not os.path.exists(args.directory):
            logging.error(f"Directory not found: {args.directory}")
            return False

        for root, _, files in os.walk(args.directory):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    files_to_process.append(file_path)

        logging.info(f"Found {len(files_to_process)} markdown files in directory: {args.directory}")
    else:
        # Process all files tracked by the document database
        logging.info("No specific files provided, processing all tracked files")
        files_to_process = orchestrator.document_db.get_all_files()

    if not files_to_process and not args.resume:
        logging.error("No files to process")
        return False

    # Run pipeline
    if args.resume:
        logging.info("Resuming from last checkpoint")
        result = await orchestrator.resume_from_checkpoint()
        if not result:
            logging.error("Failed to resume from checkpoint")
            return False
    else:
        result = await orchestrator.run(files_to_process)

    # Display results
    if result['status'] == 'completed':
        logging.info("Pipeline execution completed successfully:")
        logging.info(f"- Total files: {result['total_files']}")
        logging.info(f"- Processed files: {result['processed_files']}")
        logging.info(f"- Files with errors: {result['error_files']}")
        logging.info(f"- Elapsed time: {result['elapsed_time']:.2f} seconds")
        logging.info(f"- Throughput: {result['throughput']:.2f} files per second")
        return True
    else:
        logging.error(f"Pipeline execution failed: {result.get('error', 'Unknown error')}")
        return False

def main():
    """Main entry point of the application."""
    args = parse_args()

    # Load configuration
    config = load_config()

    # Override config with command-line arguments if provided
    if args.workers is not None:
        config['pipeline']['max_workers'] = args.workers
    if args.mode != 'auto':
        config['pipeline']['processing_mode'] = args.mode
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    if args.batch_size is not None:
        config['pipeline']['batch_size'] = args.batch_size
    if args.adaptive_scaling:
        config['pipeline']['adaptive_scaling'] = True

    # Set up logging
    setup_logging(config)

    # Initialize database
    engine, Session = init_db()
    logging.info("Database initialized")

    logging.info("Environment setup complete. Ready to process documents.")

    # Initialize document processor
    processor = DocumentProcessor(config)

    # Process files based on command line arguments
    if args.test:
        # Run test on sample file
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample.md')
        logging.info(f"Running test with sample file: {sample_path}")

        result = processor.process_file(sample_path)

        # Print summary of processing result
        print("\n--- Document Processing Test Results ---")
        print(f"File: {result['file_path']}")
        print(f"Status: {result['status']}")

        if result['status'] == 'success':
            print(f"Metadata:")
            for key, value in result['metadata'].items():
                # Skip long values
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:97]}...")
                else:
                    print(f"  {key}: {value}")

            print(f"\nChunks created: {len(result['chunks'])}")
            for i, chunk in enumerate(result['chunks']):
                print(f"\nChunk {i+1}:")
                print(f"  Section: {chunk['metadata'].get('section_title', 'N/A')}")
                print(f"  Content (first 100 chars): {chunk['content'][:100]}...")

        # Save full result to JSON for inspection
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_result.json')
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nFull result saved to: {output_path}")
    else:
        # Run pipeline
        try:
            # Run the async pipeline
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            result = asyncio.run(run_pipeline(config, args))
            sys.exit(0 if result else 1)
        except KeyboardInterrupt:
            logging.info("Pipeline execution interrupted by user")
            sys.exit(1)
        except Exception as e:
            logging.exception(f"Unexpected error: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()