"""
Command-line interface for the PKM Chatbot embedding pipeline.
"""
import argparse
import logging
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any
import tabulate
import yaml

from src.database.document_db import DocumentTracker
from src.database.checkpoint import CheckpointManager
from src.pipeline import PipelineOrchestrator


class EmbeddingPipelineCLI:
    """Command-line interface for the embedding pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the CLI with the given configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.document_db = DocumentTracker(config.get('database', {}).get('tracking_db_path'))
        self.checkpoint_manager = CheckpointManager(
            config.get('database', {}).get('checkpoint_dir', 'data/checkpoints')
        )

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the command-line argument parser."""
        parser = argparse.ArgumentParser(
            description='PKM Chatbot Embedding Pipeline CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process all documents in bulk mode
  python -m src.cli.main process --mode bulk

  # Process a single file
  python -m src.cli.main process --file path/to/file.md

  # Resume from checkpoint
  python -m src.cli.main process --resume

  # Show status of processed documents
  python -m src.cli.main status

  # Schedule processing to run daily at 2 AM
  python -m src.cli.main schedule --time "02:00"

  # Generate a unique ID
  python -m src.cli.main id --generate

  # Add IDs to all markdown files in a directory
  python -m src.cli.main id --directory path/to/dir

  # Debug configuration values
  python -m src.cli.main debug
"""
        )

        subparsers = parser.add_subparsers(dest='command', help='Command to execute')

        # Process command
        process_parser = subparsers.add_parser('process', help='Process documents')
        process_parser.add_argument('--mode', choices=['bulk', 'incremental', 'auto'], default='auto',
                            help='Processing mode: bulk, incremental, or auto')
        process_parser.add_argument('--workers', type=int, help='Number of worker processes')
        process_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
        process_parser.add_argument('--file', help='Process a single file')
        process_parser.add_argument('--directory', help='Process a directory')
        process_parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
        process_parser.add_argument('--batch-size', type=int, help='Batch size for bulk processing')
        process_parser.add_argument('--adaptive-scaling', action='store_true', help='Enable adaptive worker pool scaling')
        process_parser.add_argument('--force', action='store_true', help='Force processing of already processed files')

        # Status command
        status_parser = subparsers.add_parser('status', help='Show pipeline status')
        status_parser.add_argument('--detailed', action='store_true', help='Show detailed status')
        status_parser.add_argument('--format', choices=['table', 'json', 'csv'], default='table',
                          help='Output format for status report')
        status_parser.add_argument('--output', help='Output file for status report')
        status_parser.add_argument('--filter', choices=['all', 'completed', 'failed', 'pending'],
                          default='all', help='Filter status by document state')

        # Schedule command
        schedule_parser = subparsers.add_parser('schedule', help='Schedule pipeline execution')
        schedule_parser.add_argument('--time', help='Time to run (e.g., "02:00")')
        schedule_parser.add_argument('--interval', type=int, help='Interval in minutes')
        schedule_parser.add_argument('--remove', action='store_true', help='Remove existing schedule')
        schedule_parser.add_argument('--list', action='store_true', help='List scheduled tasks')

        # Verify command
        verify_parser = subparsers.add_parser('verify', help='Verify pipeline integrity')
        verify_parser.add_argument('--pinecone', action='store_true', help='Verify Pinecone integration')
        verify_parser.add_argument('--db', action='store_true', help='Verify document database')
        verify_parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')

        # ID command
        id_parser = subparsers.add_parser('id', help='Generate and manage document IDs')
        id_parser.add_argument('--generate', action='store_true', help='Generate a new unique ID')
        id_parser.add_argument('--file', help='Add ID to a specific markdown file')
        id_parser.add_argument('--directory', help='Add IDs to all markdown files in a directory')
        id_parser.add_argument('--force', action='store_true', help='Force ID generation even if ID exists')
        id_parser.add_argument('--format', choices=['table', 'json', 'csv'], default='table',
                       help='Output format for results')
        id_parser.add_argument('--output', help='Output file for results')

        # Debug command
        debug_parser = subparsers.add_parser('debug', help='Debug configuration values')

        return parser

    async def process_command(self, args: argparse.Namespace) -> bool:
        """Execute the process command."""
        orchestrator = PipelineOrchestrator(self.config)

        # Handle force option to reprocess files
        if args.force:
            self.logger.info("Force option enabled - will reprocess all files")
            # Reset tracking for files if they exist
            if args.file and os.path.exists(args.file):
                self.document_db.reset_file(args.file)
            elif args.directory and os.path.exists(args.directory):
                for root, _, files in os.walk(args.directory):
                    for file in files:
                        if file.endswith('.md'):
                            file_path = os.path.join(root, file)
                            self.document_db.reset_file(file_path)

        # Determine files to process
        files_to_process = []

        if args.file:
            # Process a single file
            if os.path.exists(args.file):
                files_to_process = [args.file]
            else:
                self.logger.error(f"File not found: {args.file}")
                return False
        elif args.directory:
            # Process all markdown files in a directory (recursively)
            if not os.path.exists(args.directory):
                self.logger.error(f"Directory not found: {args.directory}")
                return False

            for root, _, files in os.walk(args.directory):
                for file in files:
                    if file.endswith('.md'):
                        file_path = os.path.join(root, file)
                        files_to_process.append(file_path)

            self.logger.info(f"Found {len(files_to_process)} markdown files in directory: {args.directory}")
        else:
            # Process all files tracked by the document database
            self.logger.info("No specific files provided, processing all tracked files")
            files_to_process = self.document_db.get_all_files()

        if not files_to_process and not args.resume:
            self.logger.error("No files to process")
            return False

        # Run pipeline
        if args.resume:
            self.logger.info("Resuming from last checkpoint")
            result = await orchestrator.resume_from_checkpoint()
            if not result:
                self.logger.error("Failed to resume from checkpoint")
                return False
        else:
            result = await orchestrator.run(files_to_process)

        # Display results
        if result['status'] == 'completed':
            self.logger.info("Pipeline execution completed successfully:")
            self.logger.info(f"- Total files: {result['total_files']}")
            self.logger.info(f"- Processed files: {result['processed_files']}")
            self.logger.info(f"- Files with errors: {result['error_files']}")
            self.logger.info(f"- Elapsed time: {result['elapsed_time']:.2f} seconds")
            self.logger.info(f"- Throughput: {result['throughput']:.2f} files per second")
            return True
        else:
            self.logger.error(f"Pipeline execution failed: {result.get('error', 'Unknown error')}")
            return False

    def status_command(self, args: argparse.Namespace) -> bool:
        """Execute the status command."""
        self.logger.info("Retrieving pipeline status")

        # Get document status statistics
        stats = self.document_db.get_statistics()

        # Get document details based on filter
        if args.filter == 'all':
            documents = self.document_db.get_all_documents()
        elif args.filter == 'completed':
            documents = self.document_db.get_completed_documents()
        elif args.filter == 'failed':
            documents = self.document_db.get_error_documents()
        elif args.filter == 'pending':
            documents = self.document_db.get_pending_documents()

        # Get last checkpoint information
        checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        checkpoint_info = "None" if not checkpoint else datetime.fromtimestamp(
            checkpoint.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')

        # Display summary stats
        print("\n=== Pipeline Status Summary ===")
        print(f"Total Documents: {stats['total']}")
        print(f"Completed: {stats['completed']} ({stats['completed_percentage']:.1f}%)")
        print(f"Failed: {stats['errors']} ({stats['error_percentage']:.1f}%)")
        print(f"Pending: {stats['pending']} ({stats['pending_percentage']:.1f}%)")
        print(f"Last Checkpoint: {checkpoint_info}")

        if args.detailed and documents:
            # Prepare data for detailed view
            headers = ["File", "Status", "Last Updated", "Error"]
            rows = []

            for doc in documents:
                status = "Completed" if doc['status'] == 'completed' else "Failed" if doc['status'] == 'error' else "Pending"

                # Handle timestamp conversion - ensure it's a number before conversion
                timestamp = doc.get('timestamp')
                if isinstance(timestamp, str):
                    try:
                        # If it's an ISO format string
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                    except ValueError:
                        # If conversion fails, use current time
                        timestamp = time.time()
                elif timestamp is None:
                    timestamp = time.time()

                last_updated = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                error_msg = doc.get('error', '')
                if error_msg and len(error_msg) > 50:
                    error_msg = error_msg[:47] + "..."

                # Shorten file path for display
                file_path = doc['file_path']
                if len(file_path) > 60:
                    file_path = "..." + file_path[-57:]

                rows.append([file_path, status, last_updated, error_msg])

            # Output in requested format
            if args.format == 'table':
                print("\n=== Document Details ===")
                print(tabulate.tabulate(rows, headers=headers, tablefmt="simple"))
            elif args.format == 'json':
                output = {"documents": [{"file": row[0], "status": row[1],
                                         "last_updated": row[2], "error": row[3]} for row in rows]}
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(output, f, indent=2)
                else:
                    print(json.dumps(output, indent=2))
            elif args.format == 'csv':
                import csv
                if args.output:
                    with open(args.output, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerows(rows)
                else:
                    import io
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(headers)
                    writer.writerows(rows)
                    print(output.getvalue())

        return True

    def schedule_command(self, args: argparse.Namespace) -> bool:
        """Execute the schedule command."""
        # Check if cron is available (for Unix-like systems)
        is_windows = sys.platform.startswith('win')

        if is_windows:
            self.logger.info("Windows detected, using Task Scheduler")
            return self._schedule_windows(args)
        else:
            self.logger.info("Unix-like system detected, using cron")
            return self._schedule_unix(args)

    def _schedule_unix(self, args: argparse.Namespace) -> bool:
        """Schedule using cron on Unix-like systems."""
        try:
            import crontab

            # Get current crontab
            cron = crontab.CronTab(user=True)

            # List existing jobs
            if args.list:
                print("=== Scheduled Tasks ===")
                for job in cron:
                    if "pkm-chatbot" in str(job):
                        print(f"Command: {job.command}")
                        print(f"Schedule: {job.schedule()}")
                        print(f"Next run: {job.schedule().get_next()}")
                        print("---")
                return True

            # Remove existing schedule
            if args.remove:
                jobs_removed = 0
                for job in cron:
                    if "pkm-chatbot" in str(job):
                        cron.remove(job)
                        jobs_removed += 1

                cron.write()
                print(f"Removed {jobs_removed} scheduled tasks.")
                return True

            # Add new schedule
            if args.time:
                # Remove existing jobs first
                for job in cron:
                    if "pkm-chatbot" in str(job):
                        cron.remove(job)

                # Parse time
                try:
                    hour, minute = args.time.split(':')
                    hour, minute = int(hour), int(minute)
                except ValueError:
                    self.logger.error("Invalid time format. Use HH:MM format (e.g., 02:00)")
                    return False

                # Get absolute path to the project
                project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

                # Create new job
                job = cron.new(
                    command=f"cd {project_path} && python -m src.cli.main process --mode auto >> {project_path}/logs/scheduled_run.log 2>&1"
                )
                job.set_comment("PKM Chatbot Embedding Pipeline Scheduled Run")

                # Set schedule
                job.hour.on(hour)
                job.minute.on(minute)

                # Write to crontab
                cron.write()

                next_run = job.schedule().get_next()
                print(f"Scheduled pipeline to run at {args.time} daily")
                print(f"Next run will be at: {next_run}")

                return True

            if args.interval:
                # Remove existing jobs first
                for job in cron:
                    if "pkm-chatbot" in str(job):
                        cron.remove(job)

                # Get absolute path to the project
                project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

                # Create new job
                job = cron.new(
                    command=f"cd {project_path} && python -m src.cli.main process --mode incremental >> {project_path}/logs/scheduled_run.log 2>&1"
                )
                job.set_comment("PKM Chatbot Embedding Pipeline Interval Run")

                # Set schedule using special syntax for "every X minutes"
                job.minute.every(args.interval)

                # Write to crontab
                cron.write()

                next_run = job.schedule().get_next()
                print(f"Scheduled pipeline to run every {args.interval} minutes")
                print(f"Next run will be at: {next_run}")

                return True

            self.logger.error("No scheduling parameters provided. Use --time or --interval")
            return False

        except ImportError:
            self.logger.error("Failed to import crontab module. Install it with: pip install python-crontab")
            return False

    def _schedule_windows(self, args: argparse.Namespace) -> bool:
        """Schedule using Task Scheduler on Windows."""
        try:
            # For listing tasks
            if args.list:
                os.system('schtasks /query /fo LIST /v /tn "PKM-Chatbot*"')
                return True

            # For removing tasks
            if args.remove:
                os.system('schtasks /delete /tn "PKM-Chatbot-Daily" /f')
                print("Removed scheduled tasks.")
                return True

            # Get absolute path to the project
            project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

            # Create new task
            if args.time:
                try:
                    hour, minute = args.time.split(':')
                    hour, minute = int(hour), int(minute)
                except ValueError:
                    self.logger.error("Invalid time format. Use HH:MM format (e.g., 02:00)")
                    return False

                command = (
                    f'schtasks /create /tn "PKM-Chatbot-Daily" /tr "cmd /c cd /d {project_path} && '
                    f'python -m src.cli.main process --mode auto >> {project_path}\\logs\\scheduled_run.log 2>&1" '
                    f'/sc DAILY /st {args.time} /f'
                )

                os.system(command)
                print(f"Scheduled pipeline to run at {args.time} daily")
                return True

            if args.interval:
                command = (
                    f'schtasks /create /tn "PKM-Chatbot-Interval" /tr "cmd /c cd /d {project_path} && '
                    f'python -m src.cli.main process --mode incremental >> {project_path}\\logs\\scheduled_run.log 2>&1" '
                    f'/sc MINUTE /mo {args.interval} /f'
                )

                os.system(command)
                print(f"Scheduled pipeline to run every {args.interval} minutes")
                return True

            self.logger.error("No scheduling parameters provided. Use --time or --interval")
            return False

        except Exception as e:
            self.logger.error(f"Error scheduling task: {str(e)}")
            return False

    def verify_command(self, args: argparse.Namespace) -> bool:
        """Execute the verify command."""
        success = True

        if args.pinecone:
            # Verify Pinecone integration
            try:
                self.logger.info("Verifying Pinecone integration...")

                # Import the verification script
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
                import verify_pinecone_integration

                # Run verification
                result = verify_pinecone_integration.run_verification(self.config, fix=args.fix)

                if result['status'] == 'success':
                    print("✅ Pinecone integration verified successfully!")
                    for key, value in result['details'].items():
                        print(f"  - {key}: {value}")
                else:
                    print("❌ Pinecone integration issues found:")
                    for issue in result['issues']:
                        print(f"  - {issue}")
                    success = False

                    if args.fix and result.get('fixes_applied', []):
                        print("\nFixes applied:")
                        for fix in result['fixes_applied']:
                            print(f"  - {fix}")

            except Exception as e:
                self.logger.error(f"Error verifying Pinecone integration: {str(e)}")
                print(f"❌ Failed to verify Pinecone integration: {str(e)}")
                success = False

        if args.db:
            # Verify document database
            try:
                self.logger.info("Verifying document database...")

                # Check for inconsistencies
                inconsistencies = self.document_db.find_inconsistencies()

                if not inconsistencies:
                    print("✅ Document database integrity verified!")
                    print(f"  - Total documents: {self.document_db.get_statistics()['total']}")
                else:
                    print("❌ Document database issues found:")
                    for issue in inconsistencies:
                        print(f"  - {issue}")
                    success = False

                    # Apply fixes if requested
                    if args.fix:
                        fixed = self.document_db.repair_inconsistencies()
                        print("\nFixes applied:")
                        for fix in fixed:
                            print(f"  - {fix}")

            except Exception as e:
                self.logger.error(f"Error verifying document database: {str(e)}")
                print(f"❌ Failed to verify document database: {str(e)}")
                success = False

        return success

    def id_command(self, args: argparse.Namespace) -> bool:
        """Execute the ID command."""
        # Generate a single ID
        if args.generate:
            from src.cli.utils import generate_id
            new_id = generate_id()
            print(f"Generated ID: {new_id}")
            return True

        # Add ID to a single file
        if args.file:
            from src.cli.utils import add_id_to_markdown
            success, message = add_id_to_markdown(args.file, force=args.force)
            print(message)
            return success

        # Add IDs to all files in a directory
        if args.directory:
            from src.cli.utils import add_ids_to_directory
            results = add_ids_to_directory(args.directory, force=args.force)

            # Count successes and failures
            successes = len([r for r in results if r['success']])
            failures = len(results) - successes

            print(f"Added IDs to {successes} files in {args.directory}")
            if failures > 0:
                print(f"Failed to add IDs to {failures} files")

            # Format detailed results if requested
            if results:
                headers = ["File", "Success", "Message"]
                rows = [[os.path.basename(r['file_path']), "✅" if r['success'] else "❌", r['message']] for r in results]

                if args.format == 'table':
                    print("\n=== Detailed Results ===")
                    print(tabulate.tabulate(rows, headers=headers, tablefmt="simple"))
                elif args.format == 'json':
                    import json
                    output = {"results": [{"file": os.path.basename(r['file_path']),
                                           "success": r['success'],
                                           "message": r['message']} for r in results]}
                    if args.output:
                        with open(args.output, 'w') as f:
                            json.dump(output, f, indent=2)
                    else:
                        print(json.dumps(output, indent=2))
                elif args.format == 'csv':
                    import csv
                    if args.output:
                        with open(args.output, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)
                            writer.writerows(rows)
                    else:
                        import io
                        output = io.StringIO()
                        writer = csv.writer(output)
                        writer.writerow(headers)
                        writer.writerows(rows)
                        print(output.getvalue())

            return successes > 0 or len(results) == 0

        # If no specific action, show help
        print("Please specify an action: --generate, --file, or --directory")
        return False

    def debug_command(self, args: argparse.Namespace) -> bool:
        """
        Debug command for checking configuration settings.

        Args:
            args: Command line arguments.

        Returns:
            Dictionary with debug information.
        """
        import yaml

        # Load configuration
        config = self.config

        # Print current configuration
        print("\n===== Configuration Debug Information =====")

        # Vector Database Configuration - Corrected path
        vector_db_config = config.get('database', {}).get('vector_db', {})
        print("\nVector Database Configuration:")
        print(f"  Provider: {vector_db_config.get('provider', 'Not set')}")

        # Show API key (masked)
        api_key = vector_db_config.get('api_key', '')
        masked_key = f"{api_key[:5]}...{api_key[-4:]}" if api_key and len(api_key) > 10 else "Not set"
        print(f"  API Key: {masked_key}")

        # Show other Pinecone settings
        print(f"  Environment: {vector_db_config.get('environment', 'Not set')}")
        print(f"  Index Name: {vector_db_config.get('index_name', 'Not set')}")

        # Embedding Configuration
        embedding_config = config.get('embedding', {})
        print("\nEmbedding Configuration:")
        print(f"  Model Type: {embedding_config.get('model_type', 'Not set')}")
        print(f"  Primary Model: {embedding_config.get('primary_model', 'Not set')}")
        print(f"  Device: {embedding_config.get('device', 'Not set')}")

        # Environment Variables
        print("\nEnvironment Variables:")

        # Original raw configuration (before substitution)
        raw_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')

        try:
            with open(raw_config_path, 'r') as f:
                raw_config = yaml.safe_load(f)

            # Show raw configuration values for comparison
            print("\nRaw Configuration (Before Substitution):")

            # Check specific values in raw config
            if 'database' in raw_config and 'vector_db' in raw_config['database']:
                raw_vector_db = raw_config['database']['vector_db']
                print(f"  Raw API Key: {raw_vector_db.get('api_key', 'Not present')}")
                print(f"  Raw Environment: {raw_vector_db.get('environment', 'Not present')}")
                print(f"  Raw Index Name: {raw_vector_db.get('index_name', 'Not present')}")
        except Exception as e:
            print(f"  Error reading raw config: {str(e)}")

        # Check relevant environment variables
        env_vars = {
            'PINECONE_API_KEY': os.environ.get('PINECONE_API_KEY', 'Not set'),
            'PINECONE_ENVIRONMENT': os.environ.get('PINECONE_ENVIRONMENT', 'Not set'),
            'PINECONE_INDEX_NAME': os.environ.get('PINECONE_INDEX_NAME', 'Not set'),
            'MODEL_DEVICE': os.environ.get('MODEL_DEVICE', 'Not set')
        }

        # Mask the API key for security
        if env_vars['PINECONE_API_KEY'] != 'Not set':
            key = env_vars['PINECONE_API_KEY']
            env_vars['PINECONE_API_KEY'] = f"{key[:5]}...{key[-4:]}" if len(key) > 10 else key

        # Print environment variables
        for key, value in env_vars.items():
            print(f"  {key}: {value}")

        # Print the final configuration actually used
        print("\nFinal Configuration Values Used:")

        # Debug information about the config loading process
        import inspect
        from src import config

        print("\nConfig Module Information:")
        print(f"  Module Path: {inspect.getfile(config)}")
        print(f"  Has dotenv: {'load_dotenv' in dir(config)}")

        print("\n=========================================")

        return {"status": "success", "message": "Debug information displayed"}

    async def execute_command(self, args: argparse.Namespace) -> bool:
        """Execute the command specified in the args."""
        if args.command == 'process':
            return await self.process_command(args)
        elif args.command == 'status':
            return self.status_command(args)
        elif args.command == 'schedule':
            return self.schedule_command(args)
        elif args.command == 'verify':
            return self.verify_command(args)
        elif args.command == 'id':
            return self.id_command(args)
        elif args.command == 'debug':
            return self.debug_command(args)
        else:
            self.logger.error(f"Unknown command: {args.command}")
            return False


async def main_async(config: Dict[str, Any]) -> int:
    """Asynchronous entry point."""
    cli = EmbeddingPipelineCLI(config)
    parser = cli.create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        success = await cli.execute_command(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        logging.info("Command interrupted by user")
        return 1
    except Exception as e:
        logging.exception(f"Error executing command: {str(e)}")
        return 1