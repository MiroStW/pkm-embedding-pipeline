"""
Main document processor module that coordinates the parsing, chunking, and metadata extraction.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import frontmatter

from .document_parser import DocumentParser
from .chunking import SemanticChunker
from .metadata import MetadataExtractor

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Main document processor that coordinates parsing, chunking, and metadata extraction.

    This class serves as the entry point for document processing in the embedding pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document processor with configuration.

        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        self.parser = DocumentParser(config)
        self.chunker = SemanticChunker(config)
        self.metadata_extractor = MetadataExtractor(config)

        # Supported file extensions
        self.supported_extensions = config.get('processing', {}).get(
            'supported_extensions', ['.md', '.markdown']
        )

        logger.info(f"Initialized DocumentProcessor with supported extensions: {self.supported_extensions}")

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline.

        Args:
            file_path: Path to the file to process

        Returns:
            Dictionary containing processed document data, including metadata and chunks
        """
        if not self._is_supported_file(file_path):
            logger.warning(f"Skipping unsupported file: {file_path}")
            return {
                'status': 'skipped',
                'reason': 'unsupported_extension',
                'file_path': file_path
            }

        try:
            logger.info(f"Processing file: {file_path}")

            # Read the file and extract frontmatter
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            post = frontmatter.loads(content)
            raw_metadata = dict(post.metadata)
            raw_content = post.content

            # Extract and process metadata
            metadata = self.metadata_extractor.extract_metadata(raw_metadata, raw_content, file_path)

            # Create semantic chunks
            chunks = self.chunker.chunk_document(raw_content, metadata)

            # Update metadata with chunk count
            metadata['chunk_count'] = len(chunks)

            # Prepare result
            result = {
                'status': 'success',
                'file_path': file_path,
                'metadata': metadata,
                'chunks': chunks
            }

            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")
            return result

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

            # Return error information
            return {
                'status': 'error',
                'reason': str(e),
                'file_path': file_path
            }

    def process_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to the directory to process
            recursive: Whether to recursively process subdirectories

        Returns:
            List of processing results for each file
        """
        results = []

        try:
            logger.info(f"Processing directory: {directory_path}")

            # Walk through directory
            if recursive:
                # Process recursively
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._is_supported_file(file_path):
                            result = self.process_file(file_path)
                            results.append(result)
            else:
                # Process only current directory
                for item in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, item)
                    if os.path.isfile(file_path) and self._is_supported_file(file_path):
                        result = self.process_file(file_path)
                        results.append(result)

            logger.info(f"Processed {len(results)} files in directory {directory_path}")
            return results

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise

    def _is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is supported based on its extension.

        Args:
            file_path: Path to the file to check

        Returns:
            Boolean indicating if the file is supported
        """
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_extensions