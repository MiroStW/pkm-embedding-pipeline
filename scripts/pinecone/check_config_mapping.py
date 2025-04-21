#!/usr/bin/env python3
"""
Script to verify how the vector_db_factory is configured
and whether it's looking for the right config keys.
"""
import os
import sys
import inspect
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Check vector_db_factory configuration mapping."""
    # Load environment variables
    load_dotenv()

    # Add the project root to the path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, root_dir)

    try:
        # Import the factory
        from src.database.vector_db_factory import create_vector_db_uploader

        # Get the source code of the function
        source = inspect.getsource(create_vector_db_uploader)

        print("\n=== VECTOR DB FACTORY ANALYSIS ===")

        # Check what config keys it's looking for
        if "vector_db_config = config.get('vector_db'" in source:
            print("✅ Factory looks for 'vector_db' key")
        elif "vector_db_config = config.get('database', {}).get('vector_db'" in source:
            print("✅ Factory looks for 'database.vector_db' nested key")
        else:
            print("❓ Factory uses a different config key structure")

        # Print the full source code with line numbers for analysis
        print("\n=== VECTOR DB FACTORY SOURCE CODE ===")
        for i, line in enumerate(source.split('\n')):
            print(f"{i+1:3d}: {line}")

        # Try creating configurations with different structures
        print("\n=== TESTING DIFFERENT CONFIG STRUCTURES ===")

        api_key = os.getenv('PINECONE_API_KEY')
        environment = os.getenv('PINECONE_ENVIRONMENT')
        index_name = os.getenv('PINECONE_INDEX_NAME')

        # Test with various config structures
        test_configs = [
            {
                'vector_db': {
                    'provider': 'pinecone',
                    'api_key': api_key,
                    'environment': environment,
                    'index_name': index_name
                }
            },
            {
                'database': {
                    'vector_db': {
                        'provider': 'pinecone',
                        'api_key': api_key,
                        'environment': environment,
                        'index_name': index_name
                    }
                }
            }
        ]

        # Check the expected config structure
        for i, config in enumerate(test_configs):
            print(f"\nConfig Structure #{i+1}:")
            print(f"{config}")

            # Use a simple check without calling the function
            if 'vector_db' in config:
                print("✅ Contains 'vector_db' key (top level)")
            if 'database' in config and 'vector_db' in config['database']:
                print("✅ Contains 'database.vector_db' nested key")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())