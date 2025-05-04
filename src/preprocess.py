import argparse
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import skimage

def compare_images(img_path1, img_path2, psnr_thresh, ssim_thresh):
    """
    Compares two images using PSNR and SSIM in grayscale.

    Args:
        img_path1 (Path): Path to the first image (last kept image).
        img_path2 (Path): Path to the second image (current image).
        psnr_thresh (float): PSNR threshold for similarity.
        ssim_thresh (float): SSIM threshold for similarity.

    Returns:
        bool: True if images are similar (above both thresholds), False otherwise.
              Returns False if either image cannot be read or shapes mismatch.
    """
    try:
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Warning: Could not read image {img_path1.name} or {img_path2.name} in {img_path1.parent}. Skipping comparison.")
            return False # Treat unreadable images as dissimilar to avoid data loss

        if img1.shape != img2.shape:
            print(f"Warning: Image shapes differ {img1.shape} vs {img2.shape} for {img_path1.name} and {img_path2.name} in {img_path1.parent}. Skipping comparison.")
            return False # Treat different shapes as dissimilar

        # Calculate PSNR
        # Handle the case where images are identical (PSNR is infinite)
        try:
            # Ensure data range is appropriate for uint8 images
            psnr_val = psnr(img1, img2, data_range=img1.max() - img1.min())
        except ZeroDivisionError:
            # This happens if the images are identical
            psnr_val = float('inf')

        # Calculate SSIM
        # Ensure data range is appropriate for uint8 images
        ssim_val = ssim(img1, img2, data_range=img1.max() - img1.min())
        # print(f"Pairs: {img_path1.name} and {img_path2.name}, PSNR: {psnr_val}, SSIM: {ssim_val}")

        return psnr_val >= psnr_thresh and ssim_val >= ssim_thresh

    except Exception as e:
        print(f"Error comparing {img_path1.name} and {img_path2.name} in {img_path1.parent}: {e}")
        return False # Treat errors as dissimilar to avoid accidental data loss

def process_episodes(input_dir, output_dir, psnr_thresh, ssim_thresh):
    """
    Processes all episodes in the input directory and saves deduplicated frames
    to the output directory, maintaining the structure and renaming sequentially.

    It iterates through task directories, then episode directories within each task.
    For each episode, it compares frames sequentially against the last *kept* frame.
    If a frame is dissimilar (below thresholds) to the last kept frame, it's copied
    to the output directory with the next sequential number (0.png, 1.png, ...).
    The structure input_dir/task/episodeX -> output_dir/task/episodeX is preserved.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' not found or is not a directory.")
        return

    # Find task directories directly under the input directory
    task_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not task_dirs:
        print(f"No task directories found directly in '{input_dir}'. Check directory structure.")
        return

    print(f"Found {len(task_dirs)} potential task directories.")

    # Use tqdm for the outer loop (tasks)
    for task_dir in tqdm(task_dirs, desc="Processing Tasks", unit="task"):
        # Find episode directories within the current task directory
        # Assumes episode directories start with "episode" followed by numbers
        episode_dirs = sorted(
            [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('episode')],
            key=lambda x: int(x.name[len('episode'):]) if x.name[len('episode'):].isdigit() else -1
        )
        # Filter out any non-numeric episode names if sorting failed
        episode_dirs = [d for d in episode_dirs if d.name[len('episode'):].isdigit()]


        if not episode_dirs:
            # print(f"  No episode directories found in task '{task_dir.name}'. Skipping.")
            continue # Skip this task if no episodes found

        # print(f"\nProcessing Task: {task_dir.name} ({len(episode_dirs)} episodes)")

        # Use a nested tqdm for episodes within the current task
        for episode_dir in tqdm(episode_dirs, desc=f"  Episodes in {task_dir.name}", unit="ep", leave=False):
            output_episode_dir = output_path / task_dir.name / episode_dir.name
            # Create the output directory structure if it doesn't exist
            output_episode_dir.mkdir(parents=True, exist_ok=True)

            # Find PNG files, ensure they are named numerically, and sort them
            try:
                image_files = sorted(
                    [f for f in episode_dir.glob('*.png') if f.stem.isdigit()],
                    key=lambda p: int(p.stem)
                )
            except ValueError as e:
                print(f"    Warning: Error sorting image files in {episode_dir}. Ensure filenames are integers. Error: {e}. Skipping episode.")
                continue

            if not image_files:
                # print(f"    No numerically named PNG files found in {episode_dir}. Skipping.")
                continue # Skip this episode if no valid images found

            kept_files_count = 0
            last_kept_image_path = None # Store the *path* of the last image copied

            # --- Process Frame 0 ---
            if image_files:
                first_img_path = image_files[0]
                # Ensure the first frame name is '0.png' as expected
                if first_img_path.stem != '0':
                    print(f"    Warning: First image file in {episode_dir} is not '0.png' (found '{first_img_path.name}'). Processing may be incorrect. Skipping episode.")
                    continue

                output_img_path = output_episode_dir / f"{kept_files_count}.png"
                try:
                    # Copy the first frame unconditionally
                    shutil.copy2(first_img_path, output_img_path)
                    last_kept_image_path = first_img_path # Track the *original* path of the last kept frame for comparison
                    kept_files_count += 1
                except Exception as e:
                    print(f"    Error copying first frame {first_img_path.name} to {output_img_path}: {e}. Skipping episode.")
                    continue # Skip this episode if the first frame can't be copied

            # --- Process Subsequent Frames (Frame 1 onwards) ---
            for i in range(1, len(image_files)):
                current_image_path = image_files[i]

                # Safety check, should always have a last_kept_image_path after frame 0
                if last_kept_image_path is None:
                     print(f"    Internal Error: last_kept_image_path is None for {current_image_path.name} in {episode_dir}. Skipping frame.")
                     continue

                # Compare the current image with the *last kept* image from the *original* source
                is_similar = compare_images(last_kept_image_path, current_image_path, psnr_thresh, ssim_thresh)

                # If the image is NOT similar, keep it
                if not is_similar:
                    output_img_path = output_episode_dir / f"{kept_files_count}.png"
                    try:
                        shutil.copy2(current_image_path, output_img_path)
                        last_kept_image_path = current_image_path # Update the reference to the *original* path of the newly kept frame
                        kept_files_count += 1
                    except Exception as e:
                         print(f"    Error copying frame {current_image_path.name} to {output_img_path}: {e}. Skipping frame.")
                         # Continue processing other frames in the episode even if one fails

            # Optional: Log how many frames were kept per episode
            print(f"    Episode {episode_dir.name}: Kept {kept_files_count} / {len(image_files)} frames.")

    print("\nProcessing complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate in-episode video frames based on PSNR and SSIM similarity. "
                    "Scans input_dir/task_name/episode*/[0..N].png and writes unique frames "
                    "to output_dir/task_name/episode*/[0..M].png (where M <= N)."
    )
    parser.add_argument("input_dir", help="Directory containing the task/episode/frame.png structure.")
    parser.add_argument("output_dir", help="Directory to save the deduplicated frames, preserving the structure.")
    parser.add_argument(
        "--psnr",
        type=float,
        default=35.0,
        help="PSNR threshold for similarity. Frames with PSNR >= threshold (and SSIM >= threshold) compared to the last kept frame are dropped. Higher values mean more similarity required to drop. (Default: 35.0)"
    )
    parser.add_argument(
        "--ssim",
        type=float,
        default=0.98,
        help="SSIM threshold for similarity. Frames with SSIM >= threshold (and PSNR >= threshold) compared to the last kept frame are dropped. Value should be between 0 and 1. Higher values mean more similarity required to drop. (Default: 0.98)"
    )

    args = parser.parse_args()

    # Validate SSIM range
    if not 0.0 <= args.ssim <= 1.0:
        print("Error: SSIM threshold must be between 0.0 and 1.0.")
        return

    print("-" * 50)
    print("Starting Frame Deduplication Process")
    print(f"  Input Directory : {args.input_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  PSNR Threshold  : >= {args.psnr}")
    print(f"  SSIM Threshold  : >= {args.ssim}")
    print("-" * 50)

    process_episodes(args.input_dir, args.output_dir, args.psnr, args.ssim)

if __name__ == "__main__":
    main()
