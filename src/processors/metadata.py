"""
Metadata extraction and processing module for markdown documents.
"""
import os
import re
import logging
import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """
    Handles extraction and processing of metadata from markdown documents.

    Extracts both explicit frontmatter metadata and implicit content-based metadata.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metadata extractor with configuration.

        Args:
            config: Configuration dictionary with metadata parameters
        """
        self.config = config
        self.extract_tags = config.get('metadata', {}).get('extract_tags', True)
        self.extract_links = config.get('metadata', {}).get('extract_links', True)
        self.extract_keywords = config.get('metadata', {}).get('extract_keywords', False)

        # Regex patterns for metadata extraction
        self.tag_pattern = re.compile(r'#([a-zA-Z0-9_-]+)')
        self.link_pattern = re.compile(r'\[\[([^\]]+)\]\]')

        logger.debug(f"Initialized MetadataExtractor")

    def extract_metadata(self, frontmatter_data: Dict[str, Any], content: str, file_path: str) -> Dict[str, Any]:
        """
        Extract and process metadata from a document.

        Args:
            frontmatter_data: Raw frontmatter data from the document
            content: Document content
            file_path: Path to the document file

        Returns:
            Processed metadata dictionary
        """
        metadata = dict(frontmatter_data)

        # Basic metadata
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]

        # Ensure required fields exist
        metadata['id'] = metadata.get('id', filename_no_ext)
        metadata['title'] = metadata.get('title', filename_no_ext)
        metadata['file_path'] = file_path
        metadata['file_name'] = filename

        # Handle dates - convert string dates to datetime objects if needed
        self._process_dates(metadata)

        # Extract tags from content if enabled
        if self.extract_tags:
            content_tags = self._extract_tags_from_content(content)
            existing_tags = metadata.get('tags', [])
            if isinstance(existing_tags, str):
                existing_tags = [tag.strip() for tag in existing_tags.split(',')]

            # Combine tags from frontmatter and content
            all_tags = list(set(existing_tags + content_tags))
            metadata['tags'] = all_tags

        # Extract links if enabled
        if self.extract_links:
            metadata['links'] = self._extract_links(content)

        # Extract keywords if enabled
        if self.extract_keywords:
            metadata['keywords'] = self._extract_keywords(content)

        # Add additional metadata
        metadata['chunk_count'] = 0  # Will be updated later
        metadata['last_processed'] = datetime.datetime.now().isoformat()

        logger.debug(f"Extracted metadata for document {metadata['id']}")
        return metadata

    def _process_dates(self, metadata: Dict[str, Any]) -> None:
        """
        Process and normalize date fields in metadata.

        Args:
            metadata: Metadata dictionary to process
        """
        # Handle created date
        created = metadata.get('created')
        if created:
            if isinstance(created, str):
                try:
                    # Handle different date formats
                    if created.isdigit() or (created.startswith('"') and created[1:-1].isdigit()):
                        # Unix timestamp in milliseconds
                        timestamp = int(created.strip('"'))
                        created_date = datetime.datetime.fromtimestamp(timestamp / 1000)
                        metadata['created'] = created_date.isoformat()
                    else:
                        # Try to parse as ISO format
                        created_date = datetime.datetime.fromisoformat(created.replace('Z', '+00:00'))
                        metadata['created'] = created_date.isoformat()
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse created date: {created}")
        else:
            # Default to current time if not present
            metadata['created'] = datetime.datetime.now().isoformat()

        # Handle updated date
        updated = metadata.get('updated')
        if updated:
            if isinstance(updated, str):
                try:
                    # Handle different date formats
                    if updated.isdigit() or (updated.startswith('"') and updated[1:-1].isdigit()):
                        # Unix timestamp in milliseconds
                        timestamp = int(updated.strip('"'))
                        updated_date = datetime.datetime.fromtimestamp(timestamp / 1000)
                        metadata['updated'] = updated_date.isoformat()
                    else:
                        # Try to parse as ISO format
                        updated_date = datetime.datetime.fromisoformat(updated.replace('Z', '+00:00'))
                        metadata['updated'] = updated_date.isoformat()
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse updated date: {updated}")
        else:
            # Default to created date if not present
            metadata['updated'] = metadata['created']

    def _extract_tags_from_content(self, content: str) -> List[str]:
        """
        Extract tags from document content using regex.

        Args:
            content: Document content

        Returns:
            List of tags found in content
        """
        tags = self.tag_pattern.findall(content)
        return list(set(tags))

    def _extract_links(self, content: str) -> List[str]:
        """
        Extract internal links from document content.

        Args:
            content: Document content

        Returns:
            List of link targets found in content
        """
        links = self.link_pattern.findall(content)
        return list(set(links))

    def _extract_keywords(self, content: str) -> List[str]:
        """
        Extract potential keywords from document content.

        This is a simple implementation that could be enhanced with NLP techniques.

        Args:
            content: Document content

        Returns:
            List of potential keywords
        """
        # Simple keyword extraction based on word frequency
        # Remove code blocks
        content_without_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)

        # Remove markdown syntax
        content_clean = re.sub(r'[#*_`~\[\]\(\)\{\}]', ' ', content_without_code)

        # Split into words
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', content_clean.lower())

        # Count word frequency
        word_counts = {}
        for word in words:
            if word not in ['the', 'and', 'for', 'with', 'this', 'that', 'from']:  # Simple stopwords
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, count in sorted_words[:10] if count > 1]

        return top_keywords