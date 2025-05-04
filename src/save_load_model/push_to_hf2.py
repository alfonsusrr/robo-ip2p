import os
import argparse
from huggingface_hub import HfApi, Repository

def push_to_hf(args):
    """
    Uploads all contents of a model directory to a Hugging Face repository.
    
    Args:
        args: Command line arguments containing:
            - model_dir: Path to the model directory to upload
            - hf_repo: Name of the Hugging Face repository (format: username/repo)
            - hf_token: Hugging Face API token
    """
    # Initialize HF API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=args.hf_repo,
            token=args.hf_token,
            exist_ok=True,
            private=False
        )
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload all files in model directory
    for root, _, files in os.walk(args.model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.relpath(file_path, args.model_dir),
                    repo_id=args.hf_repo,
                    token=args.hf_token
                )
                print(f"Uploaded: {file_path}")
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Push model to Hugging Face Hub')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--hf_repo', type=str, required=True, help='HF repository name (username/repo)')
    parser.add_argument('--hf_token', type=str, required=True, help='HF API token')
    
    args = parser.parse_args()
    push_to_hf(args)
