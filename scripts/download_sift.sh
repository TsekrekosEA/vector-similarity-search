#!/bin/bash

# This script downloads and extracts the base SIFT1M dataset.
# It uses 'set -e' to exit immediately if any command fails, ensuring robustness.
set -e

echo "--- SIFT1M Dataset Download & Setup ---"

# Define file paths and URLs for clarity
DATA_DIR="data"
SIFT_ARCHIVE="$DATA_DIR/sift.tar.gz"
SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

# Create the data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Download the dataset using wget
echo "Downloading SIFT1M dataset (161MB) from $SIFT_URL..."
wget -O "$SIFT_ARCHIVE" "$SIFT_URL"

# Extract the archive directly into the data/ directory.
# The tarball contains a 'sift/' subfolder, which will be created automatically.
echo "Extracting archive..."
tar -zxvf "$SIFT_ARCHIVE" -C "$DATA_DIR/"

# Clean up the downloaded .tar.gz file, as it's no longer needed.
echo "Cleaning up downloaded archive..."
rm "$SIFT_ARCHIVE"

echo "SIFT1M dataset is ready in data/sift/"