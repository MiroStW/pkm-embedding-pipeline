"""
Checkpoint manager module for implementing checkpoint/resume functionality.
"""
import os
import json
import logging
import datetime
from typing import Dict, Any, Optional, List, Set

logger = logging.getLogger(__name__)

# Default process ID for global pipeline checkpoints
DEFAULT_PROCESS_ID = "pipeline_global"

class CheckpointManager:
    """
    Manages checkpoint saving and loading for resumable processing.
    """

    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self,
                       processed_files: Set[str] = None,
                       error_files: Set[str] = None,
                       processed_count: int = 0,
                       error_count: int = 0,
                       process_id: str = DEFAULT_PROCESS_ID) -> bool:
        """
        Save a checkpoint with the current processing state.

        Args:
            processed_files: Set of processed file paths
            error_files: Set of file paths with errors
            processed_count: Number of processed files
            error_count: Number of files with errors
            process_id: Unique identifier for the process (optional)

        Returns:
            True if checkpoint was saved successfully, False otherwise
        """
        try:
            # Convert sets to lists for JSON serialization
            state = {
                'processed_files': list(processed_files) if processed_files else [],
                'error_files': list(error_files) if error_files else [],
                'processed_count': processed_count,
                'error_count': error_count,
                'checkpoint_time': datetime.datetime.utcnow().isoformat()
            }

            # Create filename from process_id
            filename = os.path.join(self.checkpoint_dir, f"{process_id}.checkpoint.json")

            # Make sure we have a tmp_filename for atomic writing
            tmp_filename = filename + ".tmp"

            # Write to temp file first, then rename for atomicity
            with open(tmp_filename, 'w') as f:
                json.dump(state, f, indent=2)

            # Atomic rename
            os.replace(tmp_filename, filename)

            logger.info(f"Saved checkpoint for process {process_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving checkpoint for process {process_id}: {str(e)}")
            return False

    def load_checkpoint(self, process_id: str = DEFAULT_PROCESS_ID) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            process_id: Unique identifier for the process (optional)

        Returns:
            State dictionary if checkpoint exists, None otherwise
        """
        try:
            filename = os.path.join(self.checkpoint_dir, f"{process_id}.checkpoint.json")

            if not os.path.exists(filename):
                logger.info(f"No checkpoint found for process {process_id}")
                return None

            with open(filename, 'r') as f:
                state = json.load(f)

            logger.info(f"Loaded checkpoint for process {process_id} from {state.get('checkpoint_time', 'unknown time')}")
            return state

        except Exception as e:
            logger.error(f"Error loading checkpoint for process {process_id}: {str(e)}")
            return None

    def delete_checkpoint(self, process_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            process_id: Unique identifier for the process

        Returns:
            True if checkpoint was deleted successfully, False otherwise
        """
        try:
            filename = os.path.join(self.checkpoint_dir, f"{process_id}.checkpoint.json")

            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"Deleted checkpoint for process {process_id}")
                return True
            else:
                logger.warning(f"No checkpoint found to delete for process {process_id}")
                return False

        except Exception as e:
            logger.error(f"Error deleting checkpoint for process {process_id}: {str(e)}")
            return False

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their metadata.

        Returns:
            List of dictionaries with checkpoint metadata
        """
        checkpoints = []

        try:
            # Get all checkpoint files
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith(".checkpoint.json"):
                    try:
                        filepath = os.path.join(self.checkpoint_dir, filename)
                        process_id = filename.split('.')[0]

                        # Get file stats
                        stats = os.stat(filepath)
                        file_size = stats.st_size
                        modified_time = datetime.datetime.fromtimestamp(stats.st_mtime)

                        # Get basic info from the checkpoint
                        with open(filepath, 'r') as f:
                            data = json.load(f)

                        checkpoint_info = {
                            'process_id': process_id,
                            'modified_time': modified_time.isoformat(),
                            'file_size': file_size,
                            'checkpoint_time': data.get('checkpoint_time', 'unknown')
                        }

                        # Add some summary stats if available
                        if 'processed_count' in data:
                            checkpoint_info['processed_count'] = data['processed_count']
                        if 'total_count' in data:
                            checkpoint_info['total_count'] = data['total_count']

                        checkpoints.append(checkpoint_info)
                    except Exception as e:
                        logger.error(f"Error reading checkpoint {filename}: {str(e)}")

            # Sort by modified time (newest first)
            checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)
            return checkpoints

        except Exception as e:
            logger.error(f"Error listing checkpoints: {str(e)}")
            return []

    def create_bulk_processing_checkpoint(self,
                                          process_id: str,
                                          document_ids: List[str],
                                          processed_ids: Optional[List[str]] = None) -> bool:
        """
        Create a checkpoint specifically for bulk document processing.

        Args:
            process_id: Unique identifier for the bulk process
            document_ids: List of all document IDs to process
            processed_ids: List of document IDs already processed (for resume)

        Returns:
            True if checkpoint was saved successfully, False otherwise
        """
        processed_ids = processed_ids or []

        state = {
            'process_type': 'bulk_processing',
            'document_ids': document_ids,
            'processed_ids': processed_ids,
            'processed_count': len(processed_ids),
            'total_count': len(document_ids),
            'start_time': datetime.datetime.utcnow().isoformat()
        }

        return self.save_checkpoint(process_id, state)

    def update_bulk_processing_checkpoint(self,
                                          process_id: str,
                                          processed_id: str) -> bool:
        """
        Update a bulk processing checkpoint with a newly processed document ID.

        Args:
            process_id: Unique identifier for the bulk process
            processed_id: Document ID that was processed

        Returns:
            True if update was successful, False otherwise
        """
        # Load existing checkpoint
        checkpoint = self.load_checkpoint(process_id)
        if not checkpoint:
            logger.error(f"Cannot update non-existent checkpoint for process {process_id}")
            return False

        # Add processed_id to the list if it's not already there
        if processed_id not in checkpoint['processed_ids']:
            checkpoint['processed_ids'].append(processed_id)
            checkpoint['processed_count'] = len(checkpoint['processed_ids'])

        # Save updated checkpoint
        return self.save_checkpoint(process_id, checkpoint)