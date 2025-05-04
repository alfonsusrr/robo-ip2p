import os
import argparse
import torch
# Add necessary imports for Diffusers components and potentially the pipeline
import shutil
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline # Keep for potential future use or if needed by trainer
)
from transformers import CLIPTextModel, CLIPTokenizer
from train import InstructPix2PixTrainer

def save_model_components():
    parser = argparse.ArgumentParser(description='Save fine-tuned model components in Diffusers format') # Modified description
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to .ckpt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the Diffusers model') # Modified help text
    parser.add_argument('--base_model', type=str, default="timbrooks/instruct-pix2pix", help='Base model identifier used during training')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a minimal config object needed for model loading
    # Based on train.py __init__ and eval.py usage
    config = argparse.Namespace(
        base_model=args.base_model
        # Add other essential args required by __init__ if necessary,
        # though load_from_checkpoint often gets them from hparams.yaml
    )

    # Load checkpoint using our custom trainer class
    print(f"Loading checkpoint from: {args.ckpt_path}")
    # Ensure the Trainer class initializes or loads the necessary components
    # like unet, vae, text_encoder, tokenizer, scheduler
    try:
        model = InstructPix2PixTrainer.load_from_checkpoint(
            args.ckpt_path,
            map_location='cpu',
            config=config # Pass the created config object
            # It's crucial that load_from_checkpoint properly reconstructs
            # or provides access to the underlying diffusers/transformers models.
        )
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Ensure the checkpoint corresponds to the provided base_model and the Trainer class structure.")
        return # Exit if checkpoint loading fails

    # Check if the loaded model has the expected components
    # Assumes model instance has attributes like 'unet', 'vae', 'text_encoder', etc.
    components_to_save = {
        "unet": getattr(model, 'unet', None),
        "vae": getattr(model, 'vae', None),
        "text_encoder": getattr(model, 'text_encoder', None),
        "tokenizer": getattr(model, 'tokenizer', None),
        "scheduler": getattr(model, 'noise_scheduler', None) # Assuming scheduler is named noise_scheduler in Trainer
    }

    if components_to_save["unet"] is None:
         print("Error: UNet component not found in the loaded model. Cannot save.")
         return

    print(f"Saving fine-tuned components to: {args.output_dir}")

    # Save the UNet (required)
    try:
        unet_path = os.path.join(args.output_dir, "unet")
        components_to_save["unet"].save_pretrained(unet_path)
        print(f"Saved UNet to {unet_path}")
    except Exception as e:
        print(f"Error saving UNet: {e}")
        return # Stop if UNet saving fails

    # Save other components if they exist and are not None
    if components_to_save["vae"] is not None:
        try:
            vae_path = os.path.join(args.output_dir, "vae")
            components_to_save["vae"].save_pretrained(vae_path)
            print(f"Saved VAE to {vae_path}")
        except Exception as e:
            print(f"Warning: Error saving VAE: {e}")

    if components_to_save["text_encoder"] is not None:
        try:
            text_encoder_path = os.path.join(args.output_dir, "text_encoder")
            components_to_save["text_encoder"].save_pretrained(text_encoder_path)
            print(f"Saved text_encoder to {text_encoder_path}")
        except Exception as e:
            print(f"Warning: Error saving text_encoder: {e}")

    if components_to_save["tokenizer"] is not None:
        try:
            tokenizer_path = os.path.join(args.output_dir, "tokenizer")
            components_to_save["tokenizer"].save_pretrained(tokenizer_path)
            print(f"Saved tokenizer to {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Error saving tokenizer: {e}")

    if components_to_save["scheduler"] is not None:
        try:
            # Schedulers are saved using save_config to the root model directory
            scheduler_path = os.path.join(args.output_dir, "scheduler")
            components_to_save["scheduler"].save_config(scheduler_path)
            print(f"Saved scheduler config to {scheduler_path}")
        except Exception as e:
            print(f"Warning: Error saving scheduler config: {e}")

    print(f"Successfully saved fine-tuned model components in Diffusers format to: {args.output_dir}")

if __name__ == "__main__":
    save_model_components()
