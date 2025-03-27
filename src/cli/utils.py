"""
Utility functions for the PKM Chatbot CLI.
"""
import os
import uuid
import logging
import frontmatter
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def generate_id() -> str:
    """
    Generate a unique ID for a document.

    Returns:
        A globally unique ID in the format of a base36 string without vowels.
    """
    # Generate random UUID
    random_uuid = uuid.uuid4()

    # Convert to base36 (0-9, a-z)
    uuid_int = int(random_uuid.hex, 16)
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"

    # Convert to base36
    result = ""
    while uuid_int > 0:
        uuid_int, remainder = divmod(uuid_int, 36)
        result = chars[remainder] + result

    # Take first 24 characters to keep it reasonably short but still unique
    short_id = result[:24]

    # Remove vowels to prevent accidental offensive words (a, e, i, o, u)
    short_id = re.sub(r'[aeiou]', '', short_id)

    # Ensure ID is at least 16 characters
    while len(short_id) < 16:
        # Add more random characters if needed
        random_char = chars[uuid.uuid4().int % 36]
        if random_char not in 'aeiou':
            short_id += random_char

    return short_id

def add_id_to_markdown(file_path: str, force: bool = False) -> Tuple[bool, str]:
    """
    Add a unique ID to a markdown file if it doesn't already have one.

    Args:
        file_path: Path to the markdown file.
        force: If True, will generate a new ID even if one exists.

    Returns:
        Tuple of (success, message)
    """
    if not os.path.exists(file_path) or not file_path.endswith('.md'):
        return False, f"File not found or not a markdown file: {file_path}"

    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)

        # Check if already has an ID
        if 'id' in post.metadata and not force:
            return True, f"File already has ID: {post.metadata['id']}"

        # Generate new ID
        new_id = generate_id()
        post.metadata['id'] = new_id

        # Ensure created timestamp exists
        if 'created' not in post.metadata:
            post.metadata['created'] = datetime.now().strftime('%Y-%m-%d')

        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter.dumps(post))

        return True, f"Added ID {new_id} to {file_path}"

    except Exception as e:
        logger.exception(f"Error adding ID to {file_path}: {str(e)}")
        return False, f"Error adding ID: {str(e)}"

def add_ids_to_directory(directory: str, force: bool = False) -> List[Dict]:
    """
    Add IDs to all markdown files in a directory recursively.

    Args:
        directory: Path to the directory.
        force: If True, will generate new IDs even if files already have them.

    Returns:
        List of result dictionaries with file_path, success, and message.
    """
    results = []

    if not os.path.exists(directory) or not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return results

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                success, message = add_id_to_markdown(file_path, force)
                results.append({
                    'file_path': file_path,
                    'success': success,
                    'message': message
                })

    return results