#!/bin/bash
# Test script for Git hooks functionality

echo "Installing Git hooks..."
python -m src.git_hooks.cli install

echo "Checking Git hooks status..."
python -m src.git_hooks.cli status

# Test file paths
ADDED_FILE="test_added.md"
MODIFIED_FILE="README.md"
DELETED_FILE="test_delete.md"

echo "Creating a file to be deleted..."
echo "# This file will be deleted" > $DELETED_FILE
git add $DELETED_FILE
git commit -m "Add file that will be deleted"

echo "Creating a new markdown file..."
echo "# Test Added File" > $ADDED_FILE

echo "Modifying an existing file..."
echo "# Modified for testing" >> $MODIFIED_FILE

echo "Deleting the test file..."
rm $DELETED_FILE

echo "Staging all changes..."
git add .

echo "Committing changes..."
git commit -m "Test Git hooks: add, modify, delete"

echo "Checking log file for changes..."
cat logs/git_hooks.log

echo "Checking document tracker for processing queue..."
python -c "from src.database.document_db import DocumentTracker; t = DocumentTracker(); print('Files for processing:', t.get_files_for_processing()); print('Files for deletion:', t.get_files_for_deletion())"

echo "Test completed."