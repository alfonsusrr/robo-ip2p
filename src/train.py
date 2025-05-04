import os
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, CLIPTextModel

from dataset import Dataset

class InstructPix2PixTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config) # Save config to hparams

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, subfolder="tokenizer", use_fast=False)
        self.text_encoder = CLIPTextModel.from_pretrained(config.base_model, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(config.base_model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(config.base_model, subfolder="unet")

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(config.base_model, subfolder="scheduler")

        # Freeze VAE and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Ensure UNet is trainable
        self.unet.train()

        # Image logging setup
        self.image_log_counter = 0


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # --- Episode-based split --- 
            # Ensure required config values are present
            if not hasattr(self.config, 'split_episode'):

                raise ValueError("--split_episode must be provided")
            print(f"Setting up train dataset (episodes 0 to {self.config.split_episode -1})")
            self.train_dataset = Dataset(
                data_dir=self.config.data_dir, 
                frame_offset=self.config.frame_offset,
                start_episode=0,
                end_episode=self.config.split_episode
            )

            print(f"Setting up validation dataset (episodes {self.config.split_episode} to end/max)")
            self.val_dataset = Dataset(
                data_dir=self.config.data_dir, 
                frame_offset=self.config.frame_offset,
                start_episode=self.config.split_episode,
                end_episode=getattr(self.config, 'max_episodes', None) # Use max_episodes if provided
            )

            # Pre-shuffle the val dataset
            if self.val_dataset is not None:
                self.val_dataset = torch.utils.data.Subset(
                    self.val_dataset,
                    torch.randperm(len(self.val_dataset))
                )

            # --- End episode-based split --- 

            # Error checking after dataset creation
            if len(self.train_dataset) == 0:
                raise ValueError(f"Training dataset is empty. Check data_dir and episode range [0, {self.config.split_episode}).")
            if len(self.val_dataset) == 0:
                print(f"Warning: Validation dataset is empty. Check data_dir and episode range [{self.config.split_episode}, end/max). Consider adjusting split or dataset size.")
                self.val_dataset = None # Set to None if empty to avoid DataLoader issues
            else:
                print(f"Train dataset size: {len(self.train_dataset)}")
                print(f"Validation dataset size: {len(self.val_dataset)}")


    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            raise RuntimeError("train_dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        # Check if val_dataset exists and is not None after setup()
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            # Return None or an empty list if no validation data
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size, # Or a different batch size for validation
            shuffle=False, # Ensure we see the same batch index 0 every time
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        # Calculate total training steps
        num_devices = self.trainer.devices if self.trainer else 1
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader()) / self.trainer.accumulate_grad_batches) if self.trainer else len(self.train_dataloader())
        self.config.max_train_steps = self.config.epochs * num_update_steps_per_epoch


        lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.trainer.accumulate_grad_batches if self.trainer else self.config.lr_warmup_steps,
            num_training_steps=self.config.max_train_steps * self.trainer.accumulate_grad_batches if self.trainer else self.config.max_train_steps,
        )
        scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler_config]


    def forward(self, latents, timestep, encoder_hidden_states, image_latents):
        # Standard diffusion forward pass expected by diffusers UNet
        # Note: InstructPix2Pix concatenates image latents along the channel dimension
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=encoder_hidden_states).sample
        return noise_pred

    def training_step(self, batch, batch_idx):
        input_images, target_images, instructions = batch

        # Encode images to latent space (move VAE to correct device)
        # Note: No gradients needed for VAE
        with torch.no_grad():
             self.vae.to(input_images.device) # Ensure VAE is on correct device
             # mu, logvar = self.vae.encode(target_images).latent_dist # For sampling
             # latents = mu * self.vae.config.scaling_factor # Use mean for stability
             target_latents = self.vae.encode(target_images).latent_dist.sample() * self.vae.config.scaling_factor
             input_latents = self.vae.encode(input_images).latent_dist.sample() * self.vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(target_latents)
        bsz = target_latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device)
        timesteps = timesteps.long()

        # Add noise to the target latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        # Get text embeddings (move text_encoder to correct device)
        # Note: No gradients needed for text encoder
        with torch.no_grad():
            self.text_encoder.to(input_images.device) # Ensure text encoder is on correct device
            # Tokenize instructions (handle padding and truncation)
            # Ensure instructions is a list of strings
            text_inputs = self.tokenizer(
                list(instructions), padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to(input_images.device)
            encoder_hidden_states = self.text_encoder(text_input_ids)[0]


        # Predict the noise residual
        noise_pred = self(noisy_latents, timesteps, encoder_hidden_states, input_latents)


        # Calculate the loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Logging
        self.log("trainer/train_loss", loss, logger=True)

        # Log learning rate
        # lr = self.lr_schedulers().get_last_lr()[0] # Correct way to get LR from scheduler
        # self.log('lr', lr, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_images, target_images, instructions = batch

        # Similar logic as training_step, but without gradient calculation
        with torch.no_grad(): # Ensure no gradients are computed
            # Encode images
            self.vae.to(input_images.device)
            target_latents = self.vae.encode(target_images).latent_dist.sample() * self.vae.config.scaling_factor
            input_latents = self.vae.encode(input_images).latent_dist.sample() * self.vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(target_latents)
            bsz = target_latents.shape[0]

            # Sample timesteps
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device)
            timesteps = timesteps.long()

            # Add noise
            noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

            # Get text embeddings
            self.text_encoder.to(input_images.device)
            text_inputs = self.tokenizer(
                list(instructions), padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to(input_images.device)
            encoder_hidden_states = self.text_encoder(text_input_ids)[0]

            # Predict noise
            noise_pred = self(noisy_latents, timesteps, encoder_hidden_states, input_latents)

            # Calculate loss
            val_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Log validation loss
        self.log("trainer/val_loss", val_loss, logger=True)

        if self.global_rank == 0 and batch_idx == 0:
            num_save_samples = min(input_images.shape[0], 20)

            for i in range(num_save_samples):
                try:
                    # --- Prepare Tensors for Single Sample ---
                    input_img_single = input_images[i:i+1]
                    # target_img_single = target_images[i:i+1] # Not needed for saving target
                    noise_pred_single = noise_pred[i:i+1]
                    timestep_single = timesteps[i]
                    noisy_latent_single = noisy_latents[i:i+1]
                    # instruction_single = instructions[i] # Not currently used in saving

                    # --- Generate Image ---
                    pred_original_sample_single = self.noise_scheduler.step(
                        noise_pred_single, timestep_single, noisy_latent_single
                    ).pred_original_sample
                    pred_latent_to_decode = pred_original_sample_single / self.vae.config.scaling_factor
                    with torch.no_grad():
                        decoded_image = self.vae.decode(pred_latent_to_decode).sample

                    # --- Prepare Save Directory ---
                    # Unique directory for each validation run and sample
                    save_dir = os.path.join(
                        self.trainer.default_root_dir, # Use trainer's output dir
                        "validation_images",
                        f"sample_{i + 1}"
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    # --- Process and Save Generated Image ---
                    image_gen = (decoded_image[0] / 2 + 0.5).clamp(0, 1)
                    image_gen_np = image_gen.cpu().permute(1, 2, 0).numpy()
                    image_gen_uint8 = (image_gen_np * 255).round().astype("uint8")
                    pil_image_gen = Image.fromarray(image_gen_uint8)
                    save_path_gen = os.path.join(save_dir, f"generated_{self.global_step}.png")
                    pil_image_gen.save(save_path_gen)

                    # --- Process and Save Input Image --- 
                    input_img_tensor = (input_img_single[0] / 2 + 0.5).clamp(0, 1)
                    input_img_np = input_img_tensor.cpu().permute(1, 2, 0).numpy()
                    input_img_uint8 = (input_img_np * 255).round().astype("uint8")
                    pil_input_image = Image.fromarray(input_img_uint8)
                    save_path_input = os.path.join(save_dir, "input.png")
                    pil_input_image.save(save_path_input)

                    # --- Process and Save Target Image --- 
                    # Use the original target_images tensor for saving the ground truth
                    target_img_tensor = (target_images[i] / 2 + 0.5).clamp(0, 1)
                    target_img_np = target_img_tensor.cpu().permute(1, 2, 0).numpy()
                    target_img_uint8 = (target_img_np * 255).round().astype("uint8")
                    pil_target_image = Image.fromarray(target_img_uint8)
                    save_path_target = os.path.join(save_dir, "target.png")
                    pil_target_image.save(save_path_target)

                except Exception as e:
                    import traceback
                    print(f"Error saving validation sample {i} from batch {batch_idx} in epoch {self.current_epoch}: {e}")
                    traceback.print_exc()
                    continue 
        return val_loss # Return loss or other metrics if needed


def main():
    parser = argparse.ArgumentParser(description="Train InstructPix2Pix model for Robot Action Prediction")

    # --- Model & Paths ---
    parser.add_argument("--base_model", type=str, default="timbrooks/instruct-pix2pix", help="Base InstructPix2Pix model name or path")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the training data (tasks/episodes/images)")
    # parser.add_argument("--val_data_dir", type=str, default=None, help="Directory for validation data")
    parser.add_argument("--output_dir", type=str, default="./project/outputs", help="Directory to save checkpoints and logs")

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--split_episode", type=int, default=90, help="Episode to split the dataset into train and validation sets")
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional: Maximum episode index to consider for loading data overall (exclusive end).")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (-1 for all)")
    parser.add_argument("--precision", type=str, default="16", help="Training precision ('32', '16', 'bf16')") # Use '16' or 'bf16' for mixed precision
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients across batches")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_every_steps", type=int, default=500, help="Run evaluation every N steps")
    parser.add_argument("--frame_offset", type=int, default=50, help="Number of frames to offset between input and target images")

    # --- Optimizer ---
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="AdamW epsilon")


    # --- LR Scheduler ---
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="LR scheduler type (e.g., linear, cosine)")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for LR scheduler")

    # --- Wandb Logging ---
    parser.add_argument("--wandb_project", type=str, default="instruct-pix2pix-robot", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team name)")


    args = parser.parse_args()

    # --- Validation of Episode Split --- 
    if args.split_episode <= 0 or args.split_episode >= args.max_episodes:
        raise ValueError("--split_episode must be greater than 0 and less than --max_episodes")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Explicitly configure logger
    # logger = CSVLogger(save_dir=args.output_dir, name="logs")
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity, # Can be None
        save_dir=args.output_dir,
        name=f"run-{os.path.basename(args.output_dir)}-{args.base_model.split('/')[-1]}", # Example run name
    )


    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_top_k=1,  # Save only the last checkpoint
        every_n_train_steps=args.checkpoint_every_n_steps,
        save_last=False,  # Not needed since we're only saving one
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize Model
    model = InstructPix2PixTrainer(args)

    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus != 0 else "cpu",
        devices=args.gpus if args.gpus != 0 else 1,
        precision=int(args.precision), # Use precision from args
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger, # Pass the configured WandbLogger
        default_root_dir=args.output_dir,
        log_every_n_steps=50, # Log metrics every 50 steps
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.eval_every_steps # Run validation every N training steps
        # Let Trainer determine validation based on val_dataloader return
        # enable_validation=model.val_dataset is not None
        # Add strategy='ddp' if using multiple GPUs and it's not automatically inferred
        # strategy='ddp' if args.gpus > 1 else None, # Example strategy setting
    )

    # Start Training
    trainer.fit(model)


if __name__ == "__main__":
    main()
