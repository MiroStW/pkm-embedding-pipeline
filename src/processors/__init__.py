"""
Document processing components for the embedding pipeline.
"""
from .document_processor import DocumentProcessor
from .document_parser import DocumentParser
from .chunking import SemanticChunker
from .metadata import MetadataExtractor

__all__ = [
    'DocumentProcessor',
    'DocumentParser',
    'SemanticChunker',
    'MetadataExtractor'
]