#!/usr/bin/env python
"""
Verification script for resilience and recovery capabilities.

This script demonstrates recovery from simulated failures and validates data integrity.
"""

import asyncio
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    DocumentState,
    ErrorSeverity,
    PipelineStage,
    ProcessingError,
    get_logger,
    get_recovery_manager,
    init_logging,
    resilient_function,
    resilient_run,
    resilient_task,
)


# Set up logging
init_logging(log_level=10)  # 10 = DEBUG
logger = get_logger("recovery_verification")

# Unique run identifier
RECOVERY_TEST_RUN_ID = "recovery-test-run"

# Global tracking of processed documents
processed_docs = set()
fully_processed_docs = set()

# Flag file to control run sequence
FLAG_FILE = Path("./recovery_test_run.txt")


@resilient_task(stage=PipelineStage.DOCUMENT_PROCESSING, run_id=RECOVERY_TEST_RUN_ID)
async def process_document(doc_id: str, checkpoint_callback, is_first_run=True) -> dict:
    """Process a document with tracking and optional crash simulation."""
    logger.info(f"Processing document {doc_id}")

    # Simulate work
    await asyncio.sleep(0.5)

    # Track that we've processed this document
    processed_docs.add(doc_id)

    # Create document state
    doc_state = DocumentState(
        doc_id=doc_id,
        filepath=f"fake/path/{doc_id}.md"
    )
    doc_state.mark_chunks_processed()

    # Update checkpoint
    checkpoint_callback(doc_state)

    # Check if we should simulate a crash
    if is_first_run and len(processed_docs) >= 5:
        logger.warning(f"Simulating crash after processing {len(processed_docs)} documents")
        with open(FLAG_FILE, "w") as f:
            json.dump({"run": "first", "crashed": True}, f)
        os.kill(os.getpid(), signal.SIGTERM)

    # More processing
    await asyncio.sleep(0.5)
    doc_state.mark_embeddings_generated()
    checkpoint_callback(doc_state)

    # Final processing
    await asyncio.sleep(0.3)
    doc_state.mark_db_uploaded()
    checkpoint_callback(doc_state)

    # Track fully processed
    fully_processed_docs.add(doc_id)

    return {"doc_id": doc_id, "status": "completed"}


async def recovery_test():
    """Run the recovery verification test."""
    # Determine which run this is
    is_first_run = not FLAG_FILE.exists()

    if is_first_run:
        logger.info("=== FIRST RUN: WILL SIMULATE CRASH ===")
    else:
        logger.info("=== SECOND RUN: RECOVERY FROM CRASH ===")

    # Clear tracking
    processed_docs.clear()
    fully_processed_docs.clear()

    # Generate test documents
    documents = [f"doc-{i}" for i in range(10)]

    # Use a resilient run context
    with resilient_run(
        run_name="recovery-verification",
        run_id=RECOVERY_TEST_RUN_ID,
        total_documents=len(documents),
        recover_from_failure=True,  # Enable recovery
    ) as run_context:
        run_id = run_context["run_id"]
        recovery_mgr = run_context["recovery_manager"]
        checkpoint = run_context["checkpoint"]
        run_logger = run_context["logger"]

        run_logger.info(f"Starting recovery test with {len(documents)} documents")

        # Initialize document states if this is a new run
        if len(checkpoint.document_states) == 0:
            run_logger.info("Initializing document states in checkpoint")
            for doc_id in documents:
                doc_state = DocumentState(
                    doc_id=doc_id,
                    filepath=f"fake/path/{doc_id}.md"
                )
                checkpoint.add_document(doc_state)
            recovery_mgr._save_checkpoint(checkpoint)

        # Process documents, possibly with recovery logic
        recovery_state, recovered_checkpoint = recovery_mgr.get_recovery_state(run_id)

        # Check if we're recovering
        if recovery_state == "partial":
            run_logger.info("Recovering from previous partial run")

            # Use the recovered checkpoint if available
            if recovered_checkpoint:
                checkpoint = recovered_checkpoint

            # Get documents that need processing
            pending_docs = list(checkpoint.pending_documents)
            failed_docs = list(checkpoint.failed_documents)
            completed_docs = list(checkpoint.completed_documents)

            run_logger.info(f"Recovery state: {len(completed_docs)} completed, "
                            f"{len(pending_docs)} pending, {len(failed_docs)} failed")

            # Process documents that weren't fully processed
            docs_to_process = pending_docs + failed_docs

            # Also include any documents not in the checkpoint
            all_checkpoint_docs = set(pending_docs + failed_docs + completed_docs)
            docs_missing = [doc_id for doc_id in documents if doc_id not in all_checkpoint_docs]
            if docs_missing:
                run_logger.info(f"Found {len(docs_missing)} documents not in checkpoint, adding them")
                docs_to_process.extend(docs_missing)
        else:
            run_logger.info("Starting new run")
            docs_to_process = documents

        # Process each document
        for doc_id in docs_to_process:
            try:
                # Checkpoint callback
                def update_checkpoint(doc_state):
                    recovery_mgr.update_checkpoint(doc_state)

                # Process document (this may simulate a crash)
                result = await process_document(
                    doc_id,
                    update_checkpoint,
                    is_first_run=is_first_run
                )
                run_logger.info(f"Processed {doc_id}: {result}")

            except Exception as e:
                run_logger.error(f"Error processing {doc_id}: {e}")
                # Recovery happens automatically on next run

        # Generate a report
        run_logger.info(f"Completed run, processed {len(fully_processed_docs)}/{len(documents)} documents fully")

        # If this is the second run, verify all documents were processed
        if not is_first_run:
            all_processed = len(fully_processed_docs) == len(documents)
            run_logger.info(f"Recovery verification {'PASSED' if all_processed else 'FAILED'}")

            # Clean up
            get_recovery_manager().clear_recovery_data(RECOVERY_TEST_RUN_ID)
            logger.info("Cleaned up recovery data")

            # Remove the flag file
            FLAG_FILE.unlink()

            return all_processed
        else:
            # First run should have crashed, if we're here it means we didn't crash
            logger.warning("First run completed without crashing - disabled for debugging?")

            # Mark as the first run completed normally
            with open(FLAG_FILE, "w") as f:
                json.dump({"run": "first", "crashed": False}, f)

            return True


if __name__ == "__main__":
    try:
        if FLAG_FILE.exists():
            # Read flag file to determine if we're recovering from a crash
            with open(FLAG_FILE, "r") as f:
                status = json.load(f)

            if status.get("crashed", False):
                logger.info("Detected previous crash, running recovery")
            else:
                logger.info("Previous run completed normally, cleaning up")
                FLAG_FILE.unlink()
                get_recovery_manager().clear_recovery_data(RECOVERY_TEST_RUN_ID)
                sys.exit(0)
        else:
            logger.info("Starting fresh verification run")
            # Clean up any previous test run
            get_recovery_manager().clear_recovery_data(RECOVERY_TEST_RUN_ID)

        result = asyncio.run(recovery_test())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Verification interrupted")
        sys.exit(130)