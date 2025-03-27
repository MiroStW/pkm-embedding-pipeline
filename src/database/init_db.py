"""
Database initialization module for tracking documents in the embedding pipeline.
"""
import os
import yaml
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
from datetime import timezone

# Create base class for declarative models
Base = declarative_base()

class Document(Base):
    """Model representing a document in the tracking database."""
    __tablename__ = 'documents'

    id = Column(String, primary_key=True)
    filepath = Column(String, nullable=False, index=True)
    title = Column(String)
    last_modified = Column(DateTime)
    last_processed = Column(DateTime)
    status = Column(String)  # pending, processing, completed, error
    error_message = Column(Text)
    chunk_count = Column(Integer)
    embedding_model = Column(String)
    hash = Column(String)  # Content hash to detect changes

    def __repr__(self):
        return f"<Document(id='{self.id}', filepath='{self.filepath}', status='{self.status}')>"

class ProcessingQueue(Base):
    """Model representing a processing queue item."""
    __tablename__ = 'processing_queue'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, nullable=False, index=True)
    priority = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    processing_started = Column(DateTime)
    is_completed = Column(Boolean, default=False)

    def __repr__(self):
        return f"<ProcessingQueue(id={self.id}, document_id='{self.document_id}', is_completed={self.is_completed})>"

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def init_db(test_mode=False, db_path=None):
    """Initialize the database connection and create tables if they don't exist."""
    if test_mode or db_path:
        # Use the provided test database path
        db_file = db_path
    else:
        # Use the configured database path from config.yaml
        config = load_config()
        db_file = config['database']['tracking_db_path']

    # Ensure directory exists
    os.makedirs(os.path.dirname(db_file), exist_ok=True)

    # Create SQLite engine
    engine = create_engine(f"sqlite:///{db_file}")

    # Create tables
    Base.metadata.create_all(engine)

    # Create session factory
    Session = sessionmaker(bind=engine)

    return engine, Session

if __name__ == "__main__":
    # Initialize database when script is run directly
    engine, Session = init_db()
    print(f"Database initialized at {load_config()['database']['tracking_db_path']}")