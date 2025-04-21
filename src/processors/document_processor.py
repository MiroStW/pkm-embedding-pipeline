"""
Main document processor module that coordinates the parsing, chunking, and metadata extraction.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import frontmatter
import hashlib
import json

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
        print(f"DEBUG: DocumentProcessor START processing file: {file_path}")

        if not self._is_supported_file(file_path):
            logger.warning(f"Skipping unsupported file: {file_path}")
            print(f"DEBUG: DocumentProcessor SKIPPED file: {file_path}")
            return {
                'status': 'skipped',
                'reason': 'unsupported_extension',
                'file_path': file_path
            }

        try:
            logger.info(f"Processing file: {file_path}")
            logger.debug(f"File exists: {os.path.exists(file_path)}")
            logger.debug(f"File size: {os.path.getsize(file_path)} bytes")

            # Read the file and extract frontmatter
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                logger.debug(f"Successfully read file content, length: {len(content)} characters")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                print(f"DEBUG: DocumentProcessor ERRORED reading file: {file_path}. Error: {str(e)}")
                return {
                    'status': 'error',
                    'reason': f'file_read_error: {str(e)}',
                    'file_path': file_path
                }

            try:
                post = frontmatter.loads(content)
                raw_metadata = dict(post.metadata)
                raw_content = post.content
                logger.debug(f"Successfully parsed frontmatter, metadata keys: {list(raw_metadata.keys())}")
            except Exception as e:
                logger.error(f"Error parsing frontmatter in {file_path}: {str(e)}")
                print(f"DEBUG: DocumentProcessor ERRORED parsing frontmatter: {file_path}. Error: {str(e)}")
                return {
                    'status': 'error',
                    'reason': f'frontmatter_parse_error: {str(e)}',
                    'file_path': file_path
                }

            # Generate a stable document ID from the canonical path
            try:
                canonical_path = os.path.abspath(os.path.realpath(file_path))
                document_id = hashlib.sha256(canonical_path.encode('utf-8')).hexdigest()
                logger.debug(f"Generated document_id '{document_id}' for path '{canonical_path}'")
            except Exception as id_e:
                logger.error(f"Error generating document ID for {file_path}: {id_e}")
                print(f"DEBUG: DocumentProcessor ERRORED generating document ID: {file_path}. Error: {str(id_e)}")
                return {
                    'status': 'error',
                    'reason': f'document_id_generation_error: {str(id_e)}',
                    'file_path': file_path
                }

            # Extract and process metadata
            try:
                metadata = self.metadata_extractor.extract_metadata(raw_metadata, raw_content, file_path)
                # Add the generated document_id to the metadata
                metadata['document_id'] = document_id
                # Also ensure file_path is stored if not already
                if 'file_path' not in metadata:
                    metadata['file_path'] = file_path

                logger.debug(f"Successfully extracted metadata (including document_id): {metadata}")
            except Exception as e:
                logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
                print(f"DEBUG: DocumentProcessor ERRORED extracting metadata: {file_path}. Error: {str(e)}")
                return {
                    'status': 'error',
                    'reason': f'metadata_extraction_error: {str(e)}',
                    'file_path': file_path
                }

            # Create semantic chunks
            try:
                chunks = self.chunker.chunk_document(raw_content, metadata)
                logger.debug(f"Successfully created {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error creating chunks for {file_path}: {str(e)}")
                print(f"DEBUG: DocumentProcessor ERRORED creating chunks: {file_path}. Error: {str(e)}")
                return {
                    'status': 'error',
                    'reason': f'chunking_error: {str(e)}',
                    'file_path': file_path
                }

            # Update metadata with chunk count
            metadata['chunk_count'] = len(chunks)

            # Prepare result (metadata already includes document_id)
            result = {
                'status': 'success',
                'file_path': file_path,
                'metadata': metadata,
                'chunks': chunks
            }

            # DEBUG: Log the final metadata being returned
            try:
                metadata_json = json.dumps(metadata, indent=2, default=str)
                print(f"DEBUG [DocumentProcessor]: Final metadata for {file_path}:\n{metadata_json}")
            except Exception as json_e:
                print(f"DEBUG [DocumentProcessor]: Failed to serialize metadata for logging: {json_e}")

            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")
            logger.debug(f"Final result: {result}")
            print(f"DEBUG: DocumentProcessor FINISHED processing file: {file_path}. Status: {result.get('status')}")
            return result

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"DEBUG: DocumentProcessor ERRORED processing file: {file_path}. Error: {str(e)}")

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