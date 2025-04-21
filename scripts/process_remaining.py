from src.config import ConfigManager
from src.database.vector_db_factory import create_vector_db_uploader
from src.processors import DocumentProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
config = ConfigManager().load_config()
vector_db = create_vector_db_uploader(config)
processor = DocumentProcessor(config)

# Process files
files = [
    'dummy-md-files/👨‍👩‍👧‍👧-family.kids.baby-names.md',
    'dummy-md-files/logs.meetings.2024-09-12-phillip-wenig.md'
]

for file_path in files:
    logger.info(f'\nProcessing {file_path}...')
    try:
        result = processor.process_file(file_path)
        if result['status'] == 'success':
            logger.info('Processing successful, uploading to vector database...')
            vector_db.index_document(result)
            logger.info('Upload complete')
        else:
            logger.error(f'Failed to process: {result.get("error", "Unknown error")}')
    except Exception as e:
        logger.error(f'Error processing {file_path}: {str(e)}')