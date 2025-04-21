import os
from pinecone import Pinecone, ServerlessSpec
import yaml
from time import sleep

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    vector_db_config = config['database']['vector_db']

    # Initialize Pinecone
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY')
    )

    index_name = os.getenv('PINECONE_INDEX_NAME')
    dimension = vector_db_config['dimension']
    environment = os.getenv('PINECONE_ENVIRONMENT')

    print(f"Checking for existing index '{index_name}'...")

    # Delete index if it exists
    if index_name in pc.list_indexes().names():
        print(f"Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
        # Wait for the deletion to complete
        while index_name in pc.list_indexes().names():
            print("Waiting for index deletion to complete...")
            sleep(2)
        print("Index deleted successfully")

    # Create new index
    print(f"Creating new index '{index_name}' with dimension {dimension}...")

    # Create spec based on environment
    if environment == 'gcp-starter':
        # For starter (free) environment - use asia-southeast1 which is available in free tier
        spec = ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    else:
        # For other environments, use config settings
        spec = ServerlessSpec(
            cloud=vector_db_config['cloud_provider'],
            region=vector_db_config['region']
        )

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=spec
    )

    # Wait for the index to be ready
    while not index_name in pc.list_indexes().names():
        print("Waiting for index creation to complete...")
        sleep(2)

    print("Index created successfully!")
    print("\nIndex details:")
    print(pc.describe_index(index_name))

if __name__ == "__main__":
    main()