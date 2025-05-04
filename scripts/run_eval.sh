#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables for paths and parameters (makes it easier to modify)
MODEL_NAME="ip2p-robotwin-v2-10" # Or specify a local path if downloaded
MODEL_PATH="./model/${MODEL_NAME}"
# MODEL_PATH=${MODEL_NAME}
DATA_DIR="./data2"             # Relative path to the data directory
EVAL_OUTPUT_DIR="./samples/${MODEL_NAME}" # Directory to save evaluation results
SAVE_IMAGES=true  # Changed to lowercase for boolean convention
LIMIT_SAMPLES=-1 # -1 means no limit sample

BATCH_SIZE=72      # Adjust based on GPU memory for evaluation
GPUS=1              # Set to 0 for CPU, or number of GPUs to use
FRAME_OFFSET=100     # Should match the training frame offset
STEP_SIZE=10

# Episode-based splitting parameters
EVAL_START_EPISODE=90  # Start of evaluation episodes (where validation set begins)
EVAL_END_EPISODE=100   # End of evaluation episodes (optional)

# Determine device based on GPUS setting
if [ "$GPUS" -gt 0 ]; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

# Optional: Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

echo "Starting InstructPix2Pix evaluation..."
echo "Model Path: $MODEL_PATH"
echo "Data Dir: $DATA_DIR"
echo "Evaluation Output Dir: $EVAL_OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Evaluation Episodes: $EVAL_START_EPISODE to $EVAL_END_EPISODE"

python ./src/eval.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$EVAL_OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --frame_offset $FRAME_OFFSET \
    --eval_start_episode $EVAL_START_EPISODE \
    --eval_end_episode $EVAL_END_EPISODE \
    --step_size $STEP_SIZE \
    --save_images $SAVE_IMAGES \
    --limit_samples $LIMIT_SAMPLES

echo "Evaluation finished. Results saved in $EVAL_OUTPUT_DIR" 