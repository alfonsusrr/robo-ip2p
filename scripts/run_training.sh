#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables for paths and parameters (makes it easier to modify)
BASE_MODEL="timbrooks/instruct-pix2pix" # Or specify a local path if downloaded
DATA_DIR="./data"             # Relative path to the data directory
OUTPUT_DIR="./output-1x50"       # Relative path for outputs

EPOCHS=4
BATCH_SIZE=20     # Adjust based on GPU memory
LEARNING_RATE=1e-5
GPUS=1              # Set to 0 for CPU, or number of GPUs to use
PRECISION=16        # Use "32", "16", or "bf16"
NUM_WORKERS=16    # Adjust based on CPU cores
CHECKPOINT_STEPS=500
EVAL_EVERY_N_STEPS=500
FRAME_OFFSET=50
SPLIT_EPISODE=90
MAX_EPISODES=100
WANDB_PROJECT="instruct-pix2pix-robotwin"
WANDB_ENTITY="nitrous-dev"

# Optional: Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

echo "Starting InstructPix2Pix training..."
echo "Base Model: $BASE_MODEL"
echo "Data Dir: $DATA_DIR"
echo "Output Dir: $OUTPUT_DIR"

python ./src/train.py \
    --base_model "$BASE_MODEL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --gpus $GPUS \
    --precision "$PRECISION" \
    --num_workers $NUM_WORKERS \
    --checkpoint_every_n_steps $CHECKPOINT_STEPS \
    --eval_every_steps $EVAL_EVERY_N_STEPS \
    --frame_offset $FRAME_OFFSET \
    --split_episode $SPLIT_EPISODE \
    --max_episodes $MAX_EPISODES \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    | tee -a ./logs/$OUTPUT_DIR.log

echo "Training finished."
