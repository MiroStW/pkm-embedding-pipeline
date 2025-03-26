"""
Main entry point for the PKM Chatbot embedding pipeline.
"""
import os
import argparse
import logging
import yaml
import json
from src.database.init_db import init_db
from src.processors import DocumentProcessor

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

    return parser.parse_args()

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

    elif args.file:
        # Process a single file
        result = processor.process_file(args.file)
        print(f"Processed file: {args.file} - Status: {result['status']}")

    elif args.directory:
        # Process a directory
        results = processor.process_directory(args.directory)
        print(f"Processed {len(results)} files in directory: {args.directory}")

        # Print summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')

        print(f"Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    main()