#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define URLs and output names
DATA_URL="https://cloudsuite.nitrous.dev/s/nQXFPcZf5FXJ8qa/download/data.zip"
DATA2_URL="https://cloudsuite.nitrous.dev/s/7KXCx8pcZmDsBsd/download/data2.zip"
DATA_ZIP="data.zip"
DATA2_ZIP="data2.zip"

# Download and extract original data
echo "Downloading original dataset ($DATA_URL)..."
wget "$DATA_URL" -O "$DATA_ZIP"
echo "Extracting $DATA_ZIP..."
unzip -q "$DATA_ZIP" # -q for quiet extraction
echo "Removing $DATA_ZIP..."
rm "$DATA_ZIP"
echo "Original data downloaded and extracted to ./data/"

# Download and extract preprocessed data
echo "Downloading preprocessed dataset ($DATA2_URL)..."
wget "$DATA2_URL" -O "$DATA2_ZIP"
echo "Extracting $DATA2_ZIP..."
unzip -q "$DATA2_ZIP" # -q for quiet extraction
echo "Removing $DATA2_ZIP..."
rm "$DATA2_ZIP"
echo "Preprocessed data downloaded and extracted to ./data2/"

echo "Data download and extraction complete."
