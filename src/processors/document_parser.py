"""
Document parser module for extracting content and metadata from markdown files.
"""
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import frontmatter
import markdown
from markdown.blockparser import BlockParser

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    Parser for markdown documents with frontmatter metadata.

    Handles extraction of metadata, content parsing, and semantic chunking.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document parser with configuration.

        Args:
            config: Configuration dictionary with parsing parameters
        """
        self.config = config
        self.chunk_size = config.get('chunking', {}).get('max_chunk_size', 1024)
        self.chunk_overlap = config.get('chunking', {}).get('chunk_overlap', 100)
        logger.debug(f"Initialized DocumentParser with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a markdown file and extract its metadata and content.

        Args:
            file_path: Path to the markdown file

        Returns:
            Dictionary containing metadata and content
        """
        logger.debug(f"Parsing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Extract frontmatter metadata and content
            post = frontmatter.loads(content)
            metadata = dict(post.metadata)
            content = post.content

            # Extract filename without extension as a fallback title
            filename = os.path.basename(file_path)
            filename_no_ext = os.path.splitext(filename)[0]

            # Ensure required metadata fields exist
            metadata['id'] = metadata.get('id', filename_no_ext)
            metadata['title'] = metadata.get('title', filename_no_ext)
            metadata['tags'] = metadata.get('tags', [])
            metadata['created'] = metadata.get('created', None)
            metadata['updated'] = metadata.get('updated', None)
            metadata['file_path'] = file_path

            # Parse and chunk content
            chunks = self.chunk_content(content, metadata)

            return {
                'metadata': metadata,
                'content': content,
                'chunks': chunks
            }

        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise

    def chunk_content(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split content into semantic chunks based on document structure.

        Args:
            content: Markdown content to chunk
            metadata: Document metadata

        Returns:
            List of chunk dictionaries with content and metadata
        """
        # Start with basic semantic chunking based on headings
        chunks = self._create_semantic_chunks(content)

        # Enrich chunks with metadata
        for i, chunk in enumerate(chunks):
            chunk['metadata'] = metadata.copy()
            chunk['metadata']['chunk_id'] = f"{metadata['id']}_chunk_{i}"
            # Add section information if available
            if 'section_title' in chunk:
                chunk['metadata']['section_title'] = chunk['section_title']

        logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks

    def _create_semantic_chunks(self, content: str) -> List[Dict[str, Any]]:
        """
        Create semantic chunks based on document structure.

        Args:
            content: Markdown content

        Returns:
            List of chunk dictionaries
        """
        # Split by headings (# Header)
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_section_title = "Introduction"
        current_heading_level = 0

        for line in lines:
            # Check if line is a heading
            if line.strip().startswith('#'):
                # Count heading level
                heading_level = 0
                for char in line.strip():
                    if char == '#':
                        heading_level += 1
                    else:
                        break

                section_title = line.strip('#').strip()

                # If we already have content in the current chunk, save it
                if current_chunk and (heading_level <= 2 or len('\n'.join(current_chunk)) > self.chunk_size):
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section_title': current_section_title,
                        'heading_level': current_heading_level
                    })
                    current_chunk = []

                current_section_title = section_title
                current_heading_level = heading_level

            # Add line to current chunk
            current_chunk.append(line)

            # If chunk exceeds size limit, create a new chunk
            # But preserve code blocks, lists, and tables
            chunk_content = '\n'.join(current_chunk)
            if len(chunk_content) > self.chunk_size:
                # Check if we're in the middle of a code block, list, or table
                in_code_block = '```' in chunk_content and chunk_content.count('```') % 2 != 0
                in_list = self._is_in_list(current_chunk)
                in_table = '|' in chunk_content and '\n|' in chunk_content

                if not (in_code_block or in_list or in_table):
                    chunks.append({
                        'content': chunk_content,
                        'section_title': current_section_title,
                        'heading_level': current_heading_level
                    })
                    current_chunk = []

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'section_title': current_section_title,
                'heading_level': current_heading_level
            })

        return chunks

    def _is_in_list(self, lines: List[str]) -> bool:
        """
        Check if the last few lines indicate we're in the middle of a list.

        Args:
            lines: List of content lines

        Returns:
            Boolean indicating if we're in a list
        """
        if not lines:
            return False

        # Check last 3 lines for list markers
        list_markers = ['- ', '* ', '1. ', '+ ']
        for i in range(1, min(4, len(lines) + 1)):
            line = lines[-i].strip()
            if any(line.startswith(marker) for marker in list_markers):
                return True

        return False