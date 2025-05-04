import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer
import os
import math
from PIL import Image


class InstructPix2PixRoboTwin(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_load_path = getattr(config, 'model_path', self.config.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path, subfolder="tokenizer", use_fast=False)
        self.text_encoder = CLIPTextModel.from_pretrained(model_load_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_load_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_load_path, subfolder="unet")

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_load_path, subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.unet.train()

    def forward(self, latents, timestep, encoder_hidden_states, image_latents):
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=encoder_hidden_states).sample
        return noise_pred

    def training_step(self, batch, batch_idx):
        input_images, target_images, instructions = batch

        with torch.no_grad():
            self.vae.to(input_images.device)
            target_latents = self.vae.encode(target_images).latent_dist.sample() * self.vae.config.scaling_factor
            input_latents = self.vae.encode(input_images).latent_dist.sample() * self.vae.config.scaling_factor

        noise = torch.randn_like(target_latents)
        bsz = target_latents.shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        with torch.no_grad():
            self.text_encoder.to(input_images.device)
            text_inputs = self.tokenizer(
                list(instructions), padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to(input_images.device)
            encoder_hidden_states = self.text_encoder(text_input_ids)[0]

        noise_pred = self(noisy_latents, timesteps, encoder_hidden_states, input_latents)

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log("trainer/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_images, target_images, instructions = batch

        with torch.no_grad():
            self.vae.to(input_images.device)
            target_latents = self.vae.encode(target_images).latent_dist.sample() * self.vae.config.scaling_factor
            input_latents = self.vae.encode(input_images).latent_dist.sample() * self.vae.config.scaling_factor

            noise = torch.randn_like(target_latents)
            bsz = target_latents.shape[0]

            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device)
            timesteps = timesteps.long()

            noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

            self.text_encoder.to(input_images.device)
            text_inputs = self.tokenizer(
                list(instructions), padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to(input_images.device)
            encoder_hidden_states = self.text_encoder(text_input_ids)[0]

            noise_pred = self(noisy_latents, timesteps, encoder_hidden_states, input_latents)

            val_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log("trainer/val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if hasattr(self.trainer, 'logger') and self.trainer.logger and self.global_rank == 0 and batch_idx == 0 and self.trainer.is_global_zero:
            num_save_samples = min(input_images.shape[0], getattr(self.config, 'num_validation_images', 4))
            log_images = []

            for i in range(num_save_samples):
                try:
                    input_img_single = input_images[i:i+1]
                    target_img_single = target_images[i:i+1]
                    noise_pred_single = noise_pred[i:i+1]
                    timestep_single = timesteps[i]
                    noisy_latent_single = noisy_latents[i:i+1]
                    instruction_single = instructions[i]

                    pred_original_sample = self.noise_scheduler.step(
                        noise_pred_single, timestep_single, noisy_latent_single
                    ).pred_original_sample

                    pred_latent_to_decode = pred_original_sample / self.vae.config.scaling_factor
                    decoded_image = self.vae.decode(pred_latent_to_decode).sample

                    input_pil = self._tensor_to_pil(input_img_single[0])
                    target_pil = self._tensor_to_pil(target_img_single[0])
                    generated_pil = self._tensor_to_pil(decoded_image[0])

                    if isinstance(self.trainer.logger, pl.loggers.WandbLogger):
                        log_images.append(wandb.Image(input_pil, caption=f"Input {i}"))
                        log_images.append(wandb.Image(target_pil, caption=f"Target {i}"))
                        log_images.append(wandb.Image(generated_pil, caption=f"Generated {i}: {instruction_single} (Step {self.global_step})"))

                    if getattr(self.config, 'save_validation_images_locally', False):
                        save_dir = os.path.join(
                            self.trainer.default_root_dir,
                            "validation_images_model",
                            f"epoch_{self.current_epoch}_step_{self.global_step}",
                            f"sample_{i}"
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        input_pil.save(os.path.join(save_dir, "input.png"))
                        target_pil.save(os.path.join(save_dir, "target.png"))
                        generated_pil.save(os.path.join(save_dir, "generated.png"))
                        with open(os.path.join(save_dir, "instruction.txt"), "w") as f:
                            f.write(instruction_single)

                except Exception as e:
                    print(f"Error logging/saving validation sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if isinstance(self.trainer.logger, pl.loggers.WandbLogger) and log_images:
                self.trainer.logger.log_image(key="validation_samples", images=log_images, step=self.global_step)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        if self.trainer and hasattr(self.trainer, 'datamodule') and self.trainer.datamodule:
            train_dataloader = self.trainer.datamodule.train_dataloader()
            if train_dataloader:
                num_devices = max(1, self.trainer.num_devices)
                effective_batch_size = self.config.batch_size * num_devices * self.trainer.accumulate_grad_batches
                num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.trainer.accumulate_grad_batches)
                max_train_steps = self.config.epochs * num_update_steps_per_epoch
            else:
                print("Warning: Could not determine max_train_steps from dataloader.")
                max_train_steps = getattr(self.config, 'max_train_steps', 10000)

        elif hasattr(self.config, 'max_train_steps'):
            max_train_steps = self.config.max_train_steps
        else:
            raise ValueError("Cannot determine max_train_steps. Ensure trainer setup or provide 'max_train_steps' in config.")

        lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.trainer.accumulate_grad_batches if self.trainer else self.config.lr_warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler_config]

    def generate(self, input_images, instructions, num_inference_steps=50, device=None, return_pil=False):
        """
        Generate output images from input images and text instructions.
        
        Args:
            input_images (torch.Tensor): Input images tensor of shape [B, C, H, W]
            instructions (list): List of string instructions
            num_inference_steps (int): Number of denoising steps
            device (str): Device to run inference on. If None, uses the model's device
            return_pil (bool): If True, returns PIL images instead of tensors
            
        Returns:
            torch.Tensor or list: Batch of output images [B, C, H, W] or list of PIL images
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        batch_size = input_images.shape[0]
        
        # Ensure model components are on the correct device
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)
        
        with torch.no_grad():
            # Encode input images
            input_latents = self.vae.encode(input_images.to(device)).latent_dist.sample() * self.vae.config.scaling_factor
                
            # Get text embeddings
            text_inputs = self.tokenizer(
                list(instructions), 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to(device)
            encoder_hidden_states = self.text_encoder(text_input_ids)[0]
                
            # Setup for inference - Sample initial noise for the batch
            latents = torch.randn(
                (batch_size, self.unet.config.in_channels // 2, input_images.shape[2] // 8, input_images.shape[3] // 8),
                device=device,
                dtype=encoder_hidden_states.dtype
            )
                
            # Set inference steps
            self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
                
            # Denoising loop
            for t in self.noise_scheduler.timesteps:
                latent_model_input = torch.cat([latents, input_latents], dim=1)
                noise_pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
                
            # Decode the image
            latents = 1 / self.vae.config.scaling_factor * latents
            predicted_images = self.vae.decode(latents).sample
            
            # Process generated images: scale to [0, 1]
            predicted_images = (predicted_images / 2 + 0.5).clamp(0, 1)
            
            if return_pil:
                # Convert to PIL images
                pil_images = []
                for i in range(batch_size):
                    pil_images.append(self._tensor_to_pil(predicted_images[i]))
                return pil_images
            
            return predicted_images
    
    def _tensor_to_pil(self, tensor):
        tensor = (tensor / 2 + 0.5).clamp(0, 1)
        img_np = tensor.cpu().permute(1, 2, 0).float().numpy()
        img_uint8 = (img_np * 255).round().astype("uint8")
        return Image.fromarray(img_uint8)


