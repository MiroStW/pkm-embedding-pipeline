"""
Semantic chunking module for markdown content.

Provides advanced chunking strategies based on document structure.
"""
import re
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Handles semantic chunking of markdown content based on document structure.

    Implements multiple chunking strategies with configurable parameters.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the semantic chunker with configuration.

        Args:
            config: Configuration dictionary with chunking parameters
        """
        self.config = config
        chunking_config = config.get('chunking', {})
        self.max_chunk_size = chunking_config.get('max_chunk_size', 1024)
        self.chunk_overlap = chunking_config.get('chunk_overlap', 100)
        self.respect_structure = chunking_config.get('respect_structure', True)
        self.preserve_code_blocks = chunking_config.get('preserve_code_blocks', True)
        self.preserve_lists = chunking_config.get('preserve_lists', True)
        self.preserve_tables = chunking_config.get('preserve_tables', True)

        # Regex patterns for structure detection
        self.heading_pattern = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```.*\n[\s\S]*?```', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|.+\|$[\s\S]*?^\|.+\|$', re.MULTILINE)

        logger.debug(f"Initialized SemanticChunker with max_size={self.max_chunk_size}, overlap={self.chunk_overlap}")

    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document based on its structure and the configured parameters.

        Args:
            content: Markdown content to chunk
            metadata: Document metadata

        Returns:
            List of chunks with content and metadata
        """
        # Get chunks based on structure
        if self.respect_structure:
            raw_chunks = self._chunk_by_structure(content)
        else:
            # Fallback to simple size-based chunking if structure is not respected
            raw_chunks = self._chunk_by_size(content)

        # Process and enrich chunks
        processed_chunks = []
        for i, chunk_data in enumerate(raw_chunks):
            chunk_content = chunk_data['content']
            chunk_metadata = metadata.copy()

            # Add chunk-specific metadata
            chunk_metadata.update({
                'chunk_id': f"{metadata.get('id', 'doc')}_chunk_{i}",
                'chunk_index': i,
                'total_chunks': len(raw_chunks)
            })

            # Add section information if available
            if 'section_title' in chunk_data:
                chunk_metadata['section_title'] = chunk_data['section_title']
            if 'heading_level' in chunk_data:
                chunk_metadata['heading_level'] = chunk_data['heading_level']

            processed_chunks.append({
                'content': chunk_content,
                'metadata': chunk_metadata
            })

        logger.debug(f"Created {len(processed_chunks)} chunks from document")
        return processed_chunks

    def _chunk_by_structure(self, content: str) -> List[Dict[str, Any]]:
        """
        Chunk content based on document structure (headings, etc.).

        Args:
            content: Markdown content

        Returns:
            List of raw chunks with content and structural metadata
        """
        # Extract document structure
        structure = self._extract_document_structure(content)

        # Create chunks based on structure
        chunks = []
        current_chunk = []
        current_section = "Introduction"
        current_level = 0

        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Check if this line starts a new section
            heading_match = self.heading_pattern.match(line) if line.strip() else None

            if heading_match:
                # Get heading level and title
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # If we have content and hit a significant heading (h1, h2),
                # or the chunk is getting too large, save current chunk
                if current_chunk and (level <= 2 or len('\n'.join(current_chunk)) > self.max_chunk_size):
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section_title': current_section,
                        'heading_level': current_level
                    })
                    current_chunk = []

                # Update section tracking
                current_section = title
                current_level = level

            # Add current line to chunk
            current_chunk.append(line)

            # Check if we should split by size while respecting structure
            if len('\n'.join(current_chunk)) > self.max_chunk_size:
                # Don't split in the middle of special structures
                if not self._in_special_structure(lines, i):
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section_title': current_section,
                        'heading_level': current_level
                    })
                    current_chunk = []

                    # Add overlap if needed
                    if self.chunk_overlap > 0:
                        # Add previous lines as context for overlap
                        overlap_start = max(0, i - self.chunk_overlap)
                        current_chunk = lines[overlap_start:i+1]

        # Add the final chunk
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'section_title': current_section,
                'heading_level': current_level
            })

        return chunks

    def _chunk_by_size(self, content: str) -> List[Dict[str, Any]]:
        """
        Chunk content based on size, with overlap.

        Args:
            content: Markdown content

        Returns:
            List of raw chunks with content
        """
        chunks = []
        lines = content.split('\n')
        current_chunk = []

        for i, line in enumerate(lines):
            current_chunk.append(line)

            # Check if we've reached max size and should create a new chunk
            if len('\n'.join(current_chunk)) > self.max_chunk_size:
                # Don't split in the middle of special structures
                if not self._in_special_structure(lines, i):
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section_title': 'Section',
                        'heading_level': 0
                    })

                    # Reset chunk but keep overlap
                    if self.chunk_overlap > 0:
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:]
                    else:
                        current_chunk = []

        # Add the final chunk if not empty
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'section_title': 'Section',
                'heading_level': 0
            })

        return chunks

    def _extract_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract the structure of a markdown document (headings hierarchy).

        Args:
            content: Markdown content

        Returns:
            List of structural elements (headings with their levels and positions)
        """
        structure = []

        # Find all headings
        for match in self.heading_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()

            structure.append({
                'level': level,
                'title': title,
                'position': position
            })

        return structure

    def _in_special_structure(self, lines: List[str], current_index: int) -> bool:
        """
        Check if the current position is within a special structure that should not be split.

        Args:
            lines: All content lines
            current_index: Current line index

        Returns:
            Boolean indicating if we're in a special structure
        """
        if not self.respect_structure:
            return False

        # Get context (few lines before and after)
        start = max(0, current_index - 5)
        end = min(len(lines), current_index + 5)
        context = '\n'.join(lines[start:end])

        # Check for code blocks
        if self.preserve_code_blocks:
            # Simple check for unclosed code blocks
            code_markers = context.count('```')
            if code_markers % 2 != 0:
                return True

        # Check for lists
        if self.preserve_lists:
            # Check if current line is part of a list
            list_markers = ['- ', '* ', '1. ', '+ ']
            current_line = lines[current_index].strip()
            prev_line = lines[current_index - 1].strip() if current_index > 0 else ""

            # Check if current line or previous line starts with a list marker
            if any(current_line.startswith(marker) for marker in list_markers) or \
               any(prev_line.startswith(marker) for marker in list_markers):
                return True

        # Check for tables
        if self.preserve_tables:
            if '|' in context and '\n|' in context:
                # Simple check for table structure
                if '|' in lines[current_index]:
                    return True

        return False