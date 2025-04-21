#!/usr/bin/env python3
"""
Script to verify our fix to the vector_db_factory.py file works with both
top-level and nested configurations.
"""
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

async def test_config_structure(structure_type, config):
    """Test a specific config structure."""
    print(f"\n=== Testing {structure_type} Configuration ===")
    print(f"Config: {config}")

    try:
        # Import the factory
        from src.database.vector_db_factory import create_vector_db_uploader
        from src.database.mock_vector_db import MockVectorDatabaseUploader

        # Create uploader
        uploader = await create_vector_db_uploader(config)

        # Check if it's a mock
        if uploader is None:
            print("❌ Factory returned None")
            return False
        elif isinstance(uploader, MockVectorDatabaseUploader):
            print("❌ Factory created a MOCK uploader")
            return False
        else:
            print(f"✅ Factory created a REAL uploader of type: {type(uploader).__name__}")

            # Check connection
            if hasattr(uploader, 'check_connection'):
                connection_result = uploader.check_connection()
                print(f"Connection test result: {connection_result}")
                if connection_result:
                    print("✅ Connection successful")
                else:
                    print("❌ Connection failed")
                    return False
            else:
                print("⚠️ Uploader doesn't have check_connection method")

            # Get stats
            if hasattr(uploader, 'get_stats'):
                stats = uploader.get_stats()
                print(f"Index stats:")
                print(f"  Dimension: {stats.get('dimension')}")
                print(f"  Vector count: {stats.get('total_vector_count')}")

            return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

async def main():
    """Main function."""
    # Load environment variables
    load_dotenv()

    # Add project root to path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, root_dir)

    # Get Pinecone configuration
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    print("\n=== PINECONE CONFIGURATION ===")
    print(f"API Key: {'Set' if api_key else 'Not set'}")
    print(f"Environment: {'Set' if environment else 'Not set'}")
    print(f"Index Name: {'Set' if index_name else 'Not set'}")

    if not all([api_key, environment, index_name]):
        print("❌ Missing required Pinecone configuration")
        return 1

    # Test top-level configuration
    top_level_config = {
        'vector_db': {
            'provider': 'pinecone',
            'api_key': api_key,
            'environment': environment,
            'index_name': index_name,
            'dimension': 1024
        }
    }
    top_level_success = await test_config_structure("Top-Level", top_level_config)

    # Test nested configuration
    nested_config = {
        'database': {
            'vector_db': {
                'provider': 'pinecone',
                'api_key': api_key,
                'environment': environment,
                'index_name': index_name,
                'dimension': 1024
            }
        }
    }
    nested_success = await test_config_structure("Nested", nested_config)

    # Report results
    print("\n=== VERIFICATION RESULTS ===")
    print(f"Top-Level Configuration: {'✅ PASS' if top_level_success else '❌ FAIL'}")
    print(f"Nested Configuration: {'✅ PASS' if nested_success else '❌ FAIL'}")

    if top_level_success and nested_success:
        print("\n✅ OVERALL RESULT: FIXED!")
        return 0
    else:
        print("\n❌ OVERALL RESULT: NOT FIXED")
        return 1

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    sys.exit(asyncio.run(main()))