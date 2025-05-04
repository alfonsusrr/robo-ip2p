from huggingface_hub import snapshot_download
import os

REPO_ID = ["alfonsusrr/ip2p-robotwin-v2-10", "alfonsusrr/ip2p-robotwin-v2-30", "alfonsusrr/ip2p-robotwin-v1-50"]
MODEL_DIR = ["./model/ip2p-robotwin-v2-10", "./model/ip2p-robotwin-v2-30", "./model/ip2p-robotwin-v1-50"]

# Download all files except checkpoint folder
for repo_id, model_dir in zip(REPO_ID, MODEL_DIR):
    print(f"Downloading {repo_id} to {model_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        ignore_patterns=["checkpoints/*", "*validation_images/*", "wandb/*"],
        local_dir_use_symlinks=False
    )