import os
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import ImageDraw, ImageFont
import multiprocessing
from functools import partial
from tqdm import tqdm
import time
from model import InstructPix2PixRoboTwin
from dataset import VideoDataset

# Add performance monitoring and profiling
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def calculate_metrics(pred_img, target_img):
    """Calculate PSNR and SSIM between prediction and target images."""
    # Ensure images are numpy arrays
    pred_img_np = np.array(pred_img)
    target_img_np = np.array(target_img)

    # Get image dimensions (height, width)
    h, w = target_img_np.shape[:2]
    min_dim = min(h, w)

    # Convert from [0, 255] uint8 to [0, 1] float
    pred_float = pred_img_np.astype(np.float32) / 255.0
    target_float = target_img_np.astype(np.float32) / 255.0
    
    # Calculate PSNR
    # Handle potential division by zero if images are identical
    if np.max(target_float) == 0 and np.max(pred_float) == 0: 
         psnr_value = float('inf') # Or handle as a perfect score case
    else:
        try:
            psnr_value = psnr(target_float, pred_float, data_range=1.0)
        except ValueError as e:
             print(f"Warning: Could not calculate PSNR for image of size {h}x{w}. Error: {e}")
             psnr_value = 0.0 # Or np.nan
    
    # Calculate SSIM
    # Determine appropriate win_size (must be odd and <= min_dim)
    # Default win_size is 7, use min(7, min_dim) if min_dim < 7
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1 # Ensure it's odd

    # Ensure win_size is at least 3 for SSIM calculation if possible
    win_size = max(3, win_size)

    ssim_value = 0.0 # Default value
    if win_size >= 3: # SSIM requires a minimum window size
        try:
            ssim_value = ssim(
                target_float, 
                pred_float, 
                win_size=win_size, 
                channel_axis=2, # Specify channel axis for multichannel images
                data_range=1.0
            )
        except ValueError as e:
            print(f"Warning: Could not calculate SSIM for image of size {h}x{w} with win_size={win_size}. Error: {e}")
            ssim_value = 0.0 # Or np.nan
    else:
         print(f"Warning: Skipping SSIM calculation for image of size {h}x{w} because minimum dimension {min_dim} is too small for a valid window size.")

    return psnr_value, ssim_value

def load_model(config, device="cuda"):
    """Load the model from the specified directory."""
    model = InstructPix2PixRoboTwin(config)
    model.to(device)
    model.eval()
    return model

def save_visualization_worker(args_tuple):
    """
    Worker function to create and save the visualization grid for one sample.
    """
    (
        input_pil, 
        pred_pils, 
        gt_pils, 
        output_path, 
        labels, 
        step_size
    ) = args_tuple

    num_cols = len(pred_pils) + 1 # Input + Preds
    if num_cols != len(gt_pils) + 1:
         print(f"Warning (worker): Mismatch in number of prediction ({num_cols-1}) and GT ({len(gt_pils)}) images for saving {output_path}.")
         return # Avoid saving incorrect image

    width, height = input_pil.size
    
    # Row 1: Input + Predicted Frames
    row1_images = [input_pil] + pred_pils
    
    # Row 2: Blank + Ground Truth Frames
    row2_images = [Image.new('RGB', (width, height), color=(255, 255, 255))] + gt_pils # Add blank placeholder

    # Create combined image
    label_height = 30 
    row_height = height
    combined_width = width * num_cols
    combined_height = label_height + (row_height * 2)
    
    combined_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined_image)

    # Paste images
    for idx, pil_img in enumerate(row1_images):
        combined_image.paste(pil_img, (idx * width, label_height))
    for idx, pil_img in enumerate(row2_images):
        combined_image.paste(pil_img, (idx * width, label_height + row_height))
        
    # Add labels
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        
    # Use pre-defined labels passed in args_tuple
    for idx, label in enumerate(labels):
        text_width = draw.textlength(label, font=font)
        draw.text((idx * width + (width - text_width) // 2, 5), label, fill=(0, 0, 0), font=font)

    # Save
    try:
        combined_image.save(output_path)
    except Exception as e:
        print(f"Error saving visualization {output_path}: {e}")

def calculate_metrics_worker(args_tuple):
    """
    Worker function to calculate PSNR and SSIM for one pair of images.
    """
    pred_uint8, gt_uint8 = args_tuple
    
    # Ensure images are numpy arrays
    pred_np = np.array(pred_uint8)
    gt_np = np.array(gt_uint8)

    # Remove leading batch dimension if present
    if pred_np.ndim == 4 and pred_np.shape[0] == 1:
        pred_np = pred_np.squeeze(0)
    if gt_np.ndim == 4 and gt_np.shape[0] == 1:
        gt_np = gt_np.squeeze(0)
        
    # Basic check for compatibility after potential squeeze
    if pred_np.shape != gt_np.shape:
        print(f"Warning (metrics worker): Shape mismatch after squeeze - Pred: {pred_np.shape}, GT: {gt_np.shape}. Skipping metrics.")
        return 0.0, 0.0

    # Calculate metrics logic
    h, w = gt_np.shape[:2]
    min_dim = min(h, w)

    pred_float = pred_np.astype(np.float32) / 255.0
    target_float = gt_np.astype(np.float32) / 255.0
    
    psnr_value = 0.0
    if np.max(target_float) == 0 and np.max(pred_float) == 0: 
         psnr_value = float('inf') 
    else:
        try:
            psnr_value = psnr(target_float, pred_float, data_range=1.0) 
        except ValueError as e:
             psnr_value = 0.0 
    
    ssim_value = 0.0
    win_size = min(7, min_dim)
    if win_size % 2 == 0: win_size -= 1
    win_size = max(3, win_size)

    if win_size >= 3: 
        try:
            ssim_value = ssim(
                target_float, 
                pred_float, 
                win_size=win_size, 
                channel_axis=2, 
                data_range=1.0
            )
        except ValueError as e:
            ssim_value = 0.0 

    return psnr_value, ssim_value

@time_function
def generate_autoregressive_predictions(model, dataloader, output_dir, device, step_size=10, num_workers=None, save_images=True, use_fp16=True, inference_steps=50):
    """Generate predictions autoregressively and calculate metrics for each step."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of workers for multiprocessing
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)
    
    num_steps = model.config.frame_offset // step_size
    
    # Lists to hold all step metrics across all samples
    all_step_psnr = []
    all_step_ssim = []

    file_path = os.path.join(output_dir, "metrics.txt")
    f = open(file_path, "w")
    f.write(f"Autoregressive Prediction - Average Metrics per Step:\n")
    
    # Enable automatic mixed precision for faster inference if supported
    amp_dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32
    
    # Init multiprocessing pool for metric calculation and visualization
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Record generation time per batch
        total_generation_time = 0
        total_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch_start_time = time.time()
            
            # Unpack batch: input_images [B,C,H,W], gt_image_list (list of [B,C,H,W]), instructions (list of strings)
            input_images, gt_image_list, instructions = batch
            batch_size = input_images.shape[0]
            
            # Move input tensors to device
            input_images = input_images.to(device)
            gt_tensors_device = [gt.to(device) for gt in gt_image_list]
            
            # Track intermediate predictions for all samples in batch
            batch_autoregressive_preds = []
            
            # Generate all steps at once to reduce overhead
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_fp16):
                # First generation
                start_time = time.time()
                
                # Generate initial predictions (i → i+10) for whole batch
                current_input_batch = input_images
                # Use model's generate method
                current_preds = model.generate(current_input_batch, instructions, device=device, num_inference_steps=inference_steps)
                batch_autoregressive_preds.append(current_preds.detach().cpu())  # Store CPU tensor
                
                first_step_time = time.time() - start_time
                print(f"Batch {batch_idx}: First generation step took {first_step_time:.2f} seconds")
                
                # Continue autoregressive prediction for remaining steps
                for step in range(1, num_steps):
                    step_start = time.time()
                    
                    # Use previous predictions as next input
                    current_input_batch = current_preds
                    # Use model's generate method
                    current_preds = model.generate(current_input_batch, instructions, device=device, num_inference_steps=inference_steps)
                    batch_autoregressive_preds.append(current_preds.detach().cpu())  # Store CPU tensor
                    
                    step_time = time.time() - step_start
                    print(f"Batch {batch_idx}: Generation step {step+1} took {step_time:.2f} seconds")
            
            # Process each sample in the batch in parallel
            batch_metrics_args = []
            batch_visualization_args = []
            
            for sample_idx in range(batch_size):
                # Extract predictions and ground truth for this sample
                sample_predictions = [preds[sample_idx] for preds in batch_autoregressive_preds]
                sample_gt = [gt[sample_idx].cpu() for gt in gt_tensors_device]
                
                # Convert tensors to numpy for metrics and visualization
                # Predictions: convert to numpy arrays (scale to 0-255 uint8)
                pred_uint8_list = []
                for pred_tensor in sample_predictions:
                    pred_np = pred_tensor.permute(1, 2, 0).numpy()
                    pred_uint8 = (pred_np * 255).round().astype("uint8")
                    pred_uint8_list.append(pred_uint8)
                
                # Ground truth: convert to numpy arrays (scale to 0-255 uint8)
                gt_uint8_list = []
                for gt_tensor in sample_gt:
                    gt_tensor_norm = (gt_tensor / 2 + 0.5).clamp(0, 1)
                    gt_np = gt_tensor_norm.permute(1, 2, 0).numpy()
                    gt_uint8 = (gt_np * 255).round().astype("uint8")
                    gt_uint8_list.append(gt_uint8)
                
                # Prepare metric calculation args for this sample
                metric_args = [(pred_uint8_list[k], gt_uint8_list[k]) for k in range(num_steps)]
                batch_metrics_args.extend(metric_args)
                
                # Prepare visualization in parallel
                if save_images:
                    # Convert input image for visualization
                    input_np_uint8 = ((input_images[sample_idx].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
                    input_pil = Image.fromarray(input_np_uint8).convert('RGB')
                    
                    # Convert predictions and ground truth to PIL images
                    pred_pils = [Image.fromarray(pred_uint8).convert('RGB') for pred_uint8 in pred_uint8_list]
                    gt_pils = [Image.fromarray(gt_uint8).convert('RGB') for gt_uint8 in gt_uint8_list]
                    
                    # Create labels
                    labels = ["Input (i)"] + [f"Frame (i+{(s+1)*step_size})" for s in range(num_steps)]
                    
                    # Create output path for this sample
                    viz_path = os.path.join(output_dir, f"{batch_idx * batch_size + sample_idx}.png")
                    
                    # Submit visualization job
                    viz_args = (input_pil, pred_pils, gt_pils, viz_path, labels, step_size)
                    batch_visualization_args.append(viz_args)
            
            # Run all metric calculations for this batch in parallel
            start_metrics_time = time.time()
            metrics_results = pool.map(calculate_metrics_worker, batch_metrics_args)
            metrics_time = time.time() - start_metrics_time
            print(f"Batch {batch_idx}: Metric calculation took {metrics_time:.2f} seconds")
            
            # Reorganize results by sample and step
            all_batch_psnr = []
            all_batch_ssim = []
            for sample_idx in range(batch_size):
                sample_results = metrics_results[sample_idx*num_steps:(sample_idx+1)*num_steps]
                sample_psnr = [result[0] for result in sample_results]
                sample_ssim = [result[1] for result in sample_results]
                
                all_batch_psnr.append(sample_psnr)
                all_batch_ssim.append(sample_ssim)
                
                # Print metrics for this sample
                f.write(f"Batch {batch_idx}, Sample {sample_idx}:\n")
                for k in range(num_steps):
                    frame_label = (k + 1) * step_size
                    f.write(f"  Step (i+{frame_label}): PSNR = {sample_psnr[k]:.4f}, SSIM = {sample_ssim[k]:.4f}\n")
            
            # Store metrics for all samples in this batch
            all_step_psnr.extend(all_batch_psnr)
            all_step_ssim.extend(all_batch_ssim)
            
            # Run all visualization jobs for this batch
            if save_images:
                start_viz_time = time.time()
                if batch_visualization_args:
                    for viz_args in batch_visualization_args:
                        save_visualization_worker(viz_args)
                viz_time = time.time() - start_viz_time
                print(f"Batch {batch_idx}: Visualization took {viz_time:.2f} seconds")
            
            # Record batch time
            batch_time = time.time() - batch_start_time
            total_generation_time += batch_time
            total_batches += 1
            print(f"Batch {batch_idx} completed in {batch_time:.2f} seconds")
            print(f"Average time per sample: {batch_time/batch_size:.2f} seconds")
            print(f"Average time per step: {batch_time/(batch_size*num_steps):.2f} seconds")
            print("-" * 50)
     
    # Calculate average metrics for each step
    if not all_step_psnr:
        print("Warning: No metrics were calculated. Returning empty results.")
        return None
    
    # Calculate averages across all samples
    num_samples = len(all_step_psnr)
    avg_step_psnr = [0.0] * num_steps
    avg_step_ssim = [0.0] * num_steps
    
    # Sum metrics for each step across samples
    for sample_psnr in all_step_psnr:
        for k in range(num_steps):
            avg_step_psnr[k] += sample_psnr[k]
    for sample_ssim in all_step_ssim:
        for k in range(num_steps):
            avg_step_ssim[k] += sample_ssim[k]
    
    # Divide by number of samples to get average
    avg_step_psnr = [psnr_sum / num_samples for psnr_sum in avg_step_psnr]
    avg_step_ssim = [ssim_sum / num_samples for ssim_sum in avg_step_ssim]

    # Print performance statistics
    print("\nPerformance Statistics:")
    print(f"Total generation time: {total_generation_time:.2f} seconds")
    if total_batches > 0:
        print(f"Average time per batch: {total_generation_time/total_batches:.2f} seconds")
        print(f"Average time per sample: {total_generation_time/(total_batches*batch_size):.2f} seconds")
        print(f"Average time per step: {total_generation_time/(total_batches*batch_size*num_steps):.2f} seconds")

    f.write("\nSummary of metrics:\n")
    for k in range(num_steps):
        frame_label = (k + 1) * step_size
        f.write(f"  Step (i+{frame_label}): PSNR = {avg_step_psnr[k]:.4f}, SSIM = {avg_step_ssim[k]:.4f}\n")

    f.write("\nSummary Final Frame Metrics:\n")
    f.write(f"  Final PSNR (i+{model.config.frame_offset}): {avg_step_psnr[-1]:.4f}\n")
    f.write(f"  Final SSIM (i+{model.config.frame_offset}): {avg_step_ssim[-1]:.4f}\n")
    
    f.write("\nPerformance Statistics:\n")
    f.write(f"Total generation time: {total_generation_time:.2f} seconds\n")
    if total_batches > 0:
        f.write(f"Average time per batch: {total_generation_time/total_batches:.2f} seconds\n")
        f.write(f"Average time per sample: {total_generation_time/(total_batches*batch_size):.2f} seconds\n")
        f.write(f"Average time per step: {total_generation_time/(total_batches*batch_size*num_steps):.2f} seconds\n")
    f.close()
    
    metrics = {
        'avg_step_psnr': avg_step_psnr,
        'avg_step_ssim': avg_step_ssim,
        'final_psnr': avg_step_psnr[-1],
        'final_ssim': avg_step_ssim[-1],
        'total_time': total_generation_time
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate InstructPix2PixRoboTwin model with autoregressive prediction")
    
    parser.add_argument("--model_path", type=str, default="ip2p-robotwin-v2-10", help="Directory containing the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data2", help="Directory containing the evaluation data")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--frame_offset", type=int, default=50, help="Number of frames offset for the final target image")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for autoregressive prediction and GT loading")
    parser.add_argument("--eval_start_episode", type=int, default=0, help="Index of the first episode included in the evaluation set (inclusive start)")
    parser.add_argument("--eval_end_episode", type=int, default=None, help="Optional: Index of the last episode (exclusive) to include in evaluation")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes for multiprocessing")
    parser.add_argument("--save_images", type=str, default="true", help="Whether to save images")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps for generation")
    parser.add_argument("--use_fp16", type=str, default="true", help="Whether to use FP16 for inference")
    parser.add_argument("--limit_samples", type=int, default=-1, help="Limit evaluation to first N samples (for debugging)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print settings
    print("\nEvaluation Settings:")
    print(f"Model Path: {args.model_path}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"FP16 Inference: {args.use_fp16}")
    print(f"Inference Steps: {args.num_inference_steps}")
    print(f"Frame Offset: {args.frame_offset}")
    print(f"Step Size: {args.step_size}")
    print("-" * 50)
    
    # Record start time
    total_start_time = time.time()
    
    try:
        video_dataset = VideoDataset(
            data_dir=args.data_dir, 
            frame_offset=args.frame_offset,
            step_size=args.step_size,
            start_episode=args.eval_start_episode,
            end_episode=args.eval_end_episode
        )
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return
    
    if len(video_dataset) == 0:
        print("Error: Dataset is empty. Exiting.")
        return
    
    # Limit samples if specified (for debugging)
    if args.limit_samples is not None and args.limit_samples != -1:
        # shuffle the dataset using torch.randperms
        video_dataset = torch.utils.data.Subset(video_dataset, torch.randperm(len(video_dataset))[:args.limit_samples])
        print(f"Limited evaluation to first {len(video_dataset)} samples.")
        
    eval_dataset = video_dataset 
    print(f"Evaluation dataset loaded with {len(eval_dataset)} samples from episodes {args.eval_start_episode} to {args.eval_end_episode if args.eval_end_episode else 'end'}.")
    
    dataloader_start = time.time()
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"DataLoader setup took {time.time() - dataloader_start:.2f} seconds")
    
    config = argparse.Namespace(
        model_path=args.model_path,
        frame_offset=args.frame_offset, 
        step_size=args.step_size,       
        learning_rate=1e-5, 
        lr_scheduler="cosine", 
        lr_warmup_steps=500, 
        adam_beta1=0.9, 
        adam_beta2=0.999, 
        adam_weight_decay=1e-2, 
        adam_epsilon=1e-8,
        train_end_episode=args.eval_start_episode,
        val_start_episode=args.eval_start_episode,
        val_end_episode=args.eval_end_episode,
        batch_size=args.batch_size, 
        epochs=10, 
        num_workers=4,
        num_inference_steps=args.num_inference_steps
    )
    
    model_load_start = time.time()
    try:
        model = load_model(config, args.device)
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        return
    print(f"Model loading took {time.time() - model_load_start:.2f} seconds")

    if model.config.frame_offset % args.step_size != 0:
        print(f"Error: Model's configured frame_offset ({model.config.frame_offset}) must be divisible by step_size ({args.step_size}).")
        return

    # Evaluate with multiprocessing
    metrics = generate_autoregressive_predictions(
        model, 
        eval_dataloader, 
        args.output_dir, 
        args.device,
        step_size=args.step_size,
        num_workers=args.num_workers,
        save_images=args.save_images == "true",
        use_fp16=args.use_fp16 == "true",
        inference_steps=args.num_inference_steps
    )
    
    if metrics:
        print(f"\n--- Average Evaluation Results ---")
        num_steps_eval = len(metrics['avg_step_psnr'])
        for k in range(num_steps_eval):
            frame_label = (k + 1) * args.step_size
            print(f"  Avg Step (i+{frame_label}): PSNR = {metrics['avg_step_psnr'][k]:.4f}, SSIM = {metrics['avg_step_ssim'][k]:.4f}")
        
        print(f"\nFinal Frame Metrics:")
        print(f"  Final PSNR (i+{args.frame_offset}): {metrics['final_psnr']:.4f}")
        print(f"  Final SSIM (i+{args.frame_offset}): {metrics['final_ssim']:.4f}")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal evaluation time: {total_time:.2f} seconds (including setup)")
        print(f"Total generation time: {metrics['total_time']:.2f} seconds")
    else:
        print("Evaluation did not produce metrics.")

if __name__ == "__main__":
    main()
