#!/usr/bin/env python3
"""
Script to verify that environment variables are correctly substituted in the configuration.
"""
import os
import sys
import logging
from dotenv import load_dotenv
import copy

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def main():
    """Test environment variable substitution in configuration."""
    # Add project root to path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, root_dir)

    # Load environment variables
    load_dotenv()

    # Get environment variables directly
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    print("\n=== Environment Variables Directly from .env ===")
    print(f"PINECONE_API_KEY: {'Set' if api_key else 'Not set'} (First 5 chars: {api_key[:5]}... if set)")
    print(f"PINECONE_ENVIRONMENT: {'Set' if environment else 'Not set'} (Value: {environment})")
    print(f"PINECONE_INDEX_NAME: {'Set' if index_name else 'Not set'} (Value: {index_name})")

    try:
        # Test ConfigManager
        print("\n=== Testing ConfigManager ===")
        from src.config import ConfigManager

        manager = ConfigManager()
        db_config = manager.get_database_config()

        if "vector_db" in db_config:
            vdb_config = db_config["vector_db"]
            print("\nVector DB Config from ConfigManager:")
            print(f"Provider: {vdb_config.get('provider')}")
            print(f"API Key: {'Set correctly' if vdb_config.get('api_key') == api_key else 'Error'} (First 5 chars: {vdb_config.get('api_key', '')[:5]}... if set)")
            print(f"Environment: {'Set correctly' if vdb_config.get('environment') == environment else 'Error'} (Value: {vdb_config.get('environment')})")
            print(f"Index Name: {'Set correctly' if vdb_config.get('index_name') == index_name else 'Error'} (Value: {vdb_config.get('index_name')})")
        else:
            print("No vector_db config found in database section")

        # Test the load_config function
        print("\n=== Testing load_config Function ===")
        from src.config import load_config

        config = load_config()

        if "database" in config and "vector_db" in config["database"]:
            vdb_config = config["database"]["vector_db"]
            print("\nVector DB Config from load_config:")
            print(f"Provider: {vdb_config.get('provider')}")
            print(f"API Key: {'Set correctly' if vdb_config.get('api_key') == api_key else 'Error'} (First 5 chars: {vdb_config.get('api_key', '')[:5]}... if set)")
            print(f"Environment: {'Set correctly' if vdb_config.get('environment') == environment else 'Error'} (Value: {vdb_config.get('environment')})")
            print(f"Index Name: {'Set correctly' if vdb_config.get('index_name') == index_name else 'Error'} (Value: {vdb_config.get('index_name')})")
        else:
            print("No vector_db config found in database section")

        # Test if the config is correctly passed to the vector DB factory
        print("\n=== Testing Manual Substitution ===")

        # Create a simple test config with API key, environment, and index name
        test_config = {
            "vector_db": {
                "provider": "pinecone",
                "api_key": "${PINECONE_API_KEY}",
                "environment": "${PINECONE_ENVIRONMENT}",
                "index_name": "${PINECONE_INDEX_NAME}"
            }
        }

        # Use the substitute_env_var directly
        print("\nTesting direct substitution of environment variables:")
        from src.config import substitute_env_vars

        api_key_subst = substitute_env_vars(test_config["vector_db"]["api_key"])
        env_subst = substitute_env_vars(test_config["vector_db"]["environment"])
        index_subst = substitute_env_vars(test_config["vector_db"]["index_name"])

        print(f"API Key: {'Substituted correctly' if api_key_subst == api_key else 'Error'}")
        print(f"Environment: {'Substituted correctly' if env_subst == environment else 'Error'}")
        print(f"Index Name: {'Substituted correctly' if index_subst == index_name else 'Error'}")

        # Overall result
        print("\n=== Overall Result ===")
        if (vdb_config.get('api_key') == api_key and
            vdb_config.get('environment') == environment and
            vdb_config.get('index_name') == index_name):
            print("✅ Configuration loading with environment variables is working correctly!")
        else:
            print("❌ Configuration loading has issues")

        if (api_key_subst == api_key and
            env_subst == environment and
            index_subst == index_name):
            print("✅ Direct environment variable substitution is working correctly!")
        else:
            print("❌ Direct environment variable substitution has issues")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())