#!/bin/bash

# Define variables for clarity
MODEL_NAME="ip2p-robotwin-v2-10"
CHECKPOINT_FILE="last.ckpt"
OUTPUT_DIR="./model/${MODEL_NAME}"

# Run the model saving script with clear parameters
python3 src/save_model.py \
    --ckpt_path "$CHECKPOINT_FILE" \
    --output_dir "$OUTPUT_DIR"