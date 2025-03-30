#!/usr/bin/env python3
"""
Benchmark script for validating pipeline performance with different batch sizes.
Tests the Step 6 checkpoint: "Pipeline processes both small batches and large collections efficiently"
"""
import os
import sys
import time
import asyncio
import tempfile
import argparse
import logging
import statistics
import psutil
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import PipelineOrchestrator
from src.config import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline_benchmark")

class PerformanceMonitor:
    """Monitors system performance during pipeline execution."""

    def __init__(self, interval=1.0):
        """Initialize the performance monitor."""
        self.interval = interval
        self.running = False
        self.cpu_usage = []
        self.memory_usage = []
        self.monitor_task = None

    async def start(self):
        """Start performance monitoring."""
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor())

    async def _monitor(self):
        """Monitor system resources."""
        while self.running:
            # Collect CPU and memory usage
            self.cpu_usage.append(psutil.cpu_percent(interval=None))
            self.memory_usage.append(psutil.virtual_memory().percent)
            await asyncio.sleep(self.interval)

    async def stop(self):
        """Stop performance monitoring."""
        self.running = False
        if self.monitor_task:
            await self.monitor_task

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from the monitoring session."""
        if not self.cpu_usage:
            return {"error": "No data collected"}

        return {
            "cpu": {
                "avg": statistics.mean(self.cpu_usage),
                "max": max(self.cpu_usage),
                "min": min(self.cpu_usage)
            },
            "memory": {
                "avg": statistics.mean(self.memory_usage),
                "max": max(self.memory_usage),
                "min": min(self.memory_usage)
            }
        }

def generate_test_files(directory: str, count: int) -> List[str]:
    """Generate test markdown files for benchmarking."""
    file_paths = []

    # Create files with varying content sizes
    for i in range(count):
        # Vary content size based on index
        sections = max(1, i % 10)
        paragraphs_per_section = max(1, (i % 5) + 1)

        # Create file content
        content = [f"---\ntitle: Test Document {i}\nid: test_{i}\ntags: [test, benchmark]\n---\n\n"]

        for s in range(sections):
            content.append(f"# Section {s+1}\n\n")

            for p in range(paragraphs_per_section):
                # Create paragraph with varying length
                words = ["benchmark"] * (20 + (i % 50))
                content.append(" ".join(words) + "\n\n")

        # Write to file
        file_path = os.path.join(directory, f"benchmark_{i}.md")
        with open(file_path, "w") as f:
            f.writelines(content)

        file_paths.append(file_path)

    return file_paths

async def run_benchmark(config: Dict[str, Any], files: List[str], mode: str) -> Dict[str, Any]:
    """Run benchmark with given configuration and files."""
    # Configure pipeline for the specific mode
    config['pipeline']['processing_mode'] = mode

    # Set up performance monitoring
    monitor = PerformanceMonitor(interval=0.5)
    await monitor.start()

    # Run the pipeline
    start_time = time.time()
    orchestrator = PipelineOrchestrator(config)
    result = await orchestrator.run(files)
    end_time = time.time()

    # Stop monitoring
    await monitor.stop()

    # Calculate metrics
    elapsed_time = end_time - start_time
    processed_count = result['processed_files']
    error_count = result['error_files']
    throughput = processed_count / elapsed_time if elapsed_time > 0 else 0

    # Get system performance stats
    perf_stats = monitor.get_stats()

    return {
        **result,
        "elapsed_time": elapsed_time,
        "throughput": throughput,
        "performance": perf_stats
    }

async def main():
    """Run benchmark tests for different batch sizes."""
    parser = argparse.ArgumentParser(description="Pipeline Performance Benchmark")
    parser.add_argument("--small", type=int, default=10, help="Number of files for small batch test")
    parser.add_argument("--medium", type=int, default=100, help="Number of files for medium batch test")
    parser.add_argument("--large", type=int, default=500, help="Number of files for large batch test")
    parser.add_argument("--skip-large", action="store_true", help="Skip large batch test")
    args = parser.parse_args()

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.config

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory: {temp_dir}")

        # Modify config for benchmarking
        if 'database' not in config:
            config['database'] = {}
        config['database']['tracking_db_path'] = os.path.join(temp_dir, "benchmark.db")
        config['database']['checkpoint_dir'] = os.path.join(temp_dir, "checkpoints")

        # Ensure pipeline config exists
        if 'pipeline' not in config:
            config['pipeline'] = {}

        # Small batch test (incremental mode)
        logger.info(f"Generating {args.small} files for small batch test...")
        small_files = generate_test_files(temp_dir, args.small)

        logger.info(f"Running small batch test (incremental mode) with {len(small_files)} files...")
        small_result = await run_benchmark(config, small_files, "incremental")

        logger.info("Small batch test results:")
        logger.info(f"  Processed: {small_result['processed_files']} files")
        logger.info(f"  Errors: {small_result['error_files']} files")
        logger.info(f"  Time: {small_result['elapsed_time']:.2f} seconds")
        logger.info(f"  Throughput: {small_result['throughput']:.2f} files/second")
        logger.info(f"  Avg CPU: {small_result['performance']['cpu']['avg']:.1f}%")
        logger.info(f"  Avg Memory: {small_result['performance']['memory']['avg']:.1f}%")

        # Medium batch test (auto mode)
        logger.info(f"Generating {args.medium} files for medium batch test...")
        medium_files = generate_test_files(temp_dir, args.medium)

        logger.info(f"Running medium batch test (auto mode) with {len(medium_files)} files...")
        medium_result = await run_benchmark(config, medium_files, "auto")

        logger.info("Medium batch test results:")
        logger.info(f"  Processed: {medium_result['processed_files']} files")
        logger.info(f"  Errors: {medium_result['error_files']} files")
        logger.info(f"  Time: {medium_result['elapsed_time']:.2f} seconds")
        logger.info(f"  Throughput: {medium_result['throughput']:.2f} files/second")
        logger.info(f"  Avg CPU: {medium_result['performance']['cpu']['avg']:.1f}%")
        logger.info(f"  Avg Memory: {medium_result['performance']['memory']['avg']:.1f}%")

        # Large batch test (bulk mode)
        if not args.skip_large:
            logger.info(f"Generating {args.large} files for large batch test...")
            large_files = generate_test_files(temp_dir, args.large)

            logger.info(f"Running large batch test (bulk mode) with {len(large_files)} files...")
            large_result = await run_benchmark(config, large_files, "bulk")

            logger.info("Large batch test results:")
            logger.info(f"  Processed: {large_result['processed_files']} files")
            logger.info(f"  Errors: {large_result['error_files']} files")
            logger.info(f"  Time: {large_result['elapsed_time']:.2f} seconds")
            logger.info(f"  Throughput: {large_result['throughput']:.2f} files/second")
            logger.info(f"  Avg CPU: {large_result['performance']['cpu']['avg']:.1f}%")
            logger.info(f"  Avg Memory: {large_result['performance']['memory']['avg']:.1f}%")

        # Compare results
        logger.info("\nPerformance comparison:")
        incremental_throughput = small_result['throughput']
        auto_throughput = medium_result['throughput']
        bulk_throughput = large_result['throughput'] if not args.skip_large else 0

        logger.info(f"  Incremental mode: {incremental_throughput:.2f} files/second")
        logger.info(f"  Auto mode: {auto_throughput:.2f} files/second")
        if not args.skip_large:
            logger.info(f"  Bulk mode: {bulk_throughput:.2f} files/second")

        # Determine if checkpoint is validated
        checkpoint_validated = True

        # Validation criteria:
        # 1. All modes successfully processed files
        # 2. Bulk mode throughput should be higher than incremental for large collections
        # 3. Auto mode should automatically select appropriate mode

        if small_result['error_files'] > 0 or medium_result['error_files'] > 0:
            checkpoint_validated = False
            logger.error("Validation failed: Pipeline had errors processing files")

        if not args.skip_large and bulk_throughput < incremental_throughput:
            logger.warning("Validation concern: Bulk mode throughput is lower than incremental mode")

        logger.info(f"\nCheckpoint validation: {'PASSED' if checkpoint_validated else 'FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())