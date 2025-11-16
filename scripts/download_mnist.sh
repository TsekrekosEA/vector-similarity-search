#!/bin/bash

# This script downloads and decompresses the four official example binary files
# 'set -e' ensures the script will exit immediately if any command fails.
set -e

echo "--- MNIST Dataset Download & Setup ---"

BASE_URL="https://storage.googleapis.com/tensorflow/tf-keras-datasets"
TARGET_DIR="data/mnist"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# List of the four compressed files we need to download
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

# Loop through the list, download and decompress each file
echo "Downloading 4 MNIST files from the official Google Cloud mirror..."
for FILE in "${FILES[@]}"; do
    FILE_PATH="$TARGET_DIR/$FILE"
    
    echo " -> Downloading $FILE..."
    
    # This wget command is simple and does not need a User-Agent.
    wget -O "$FILE_PATH" "$BASE_URL/$FILE"
    
    echo " -> Decompressing $FILE..."
    gunzip -f "$FILE_PATH" # Use -f to force overwrite if a previous attempt left a file
done

echo "✅ MNIST dataset is ready in $TARGET_DIR/"