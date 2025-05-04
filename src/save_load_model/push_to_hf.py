import argparse
import os
from huggingface_hub import HfApi, login

def push_to_hf():
    parser = argparse.ArgumentParser(description='Push model to Hugging Face Hub')
    parser.add_argument('--folder', type=str, required=True, help='Path to model folder to upload')
    parser.add_argument('--hf_path', type=str, required=True, help='Hugging Face repository path (e.g. username/repo-name)')
    parser.add_argument('--api_key', type=str, required=True, help='Hugging Face API key')
    
    args = parser.parse_args()

    # Login to Hugging Face Hub
    login(token=args.api_key)

    # Initialize HF API client
    api = HfApi()

    # Create repository if it doesn't exist
    api.create_repo(
        repo_id=args.hf_path,
        exist_ok=True,
        repo_type="model"
    )

    # Upload all files from folder
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.hf_path,
        repo_type="model"
    )

    print(f"Successfully uploaded model to: {args.hf_path}")

if __name__ == "__main__":
    push_to_hf()
