import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import torch

# Mapping from task directory base names to text instructions
TASK_INSTRUCTIONS = {
    "block_hammer_beat_D435": "beat the block with the hammer",
    "block_handover_D435": "handover the blocks",
    "blocks_stack_easy_D435": "stack blocks",
}

class VideoDataset(Dataset):
    def __init__(self, data_dir, frame_offset=50, step_size=10, start_episode=0, end_episode=None):
        """
        Initializes the VideoDataset.

        Args:
            data_dir (str): Path to the main data directory containing task folders.
            frame_offset (int): Number of frames offset for the final target image.
            step_size (int): Step size between intermediate frames.
            start_episode (int): The starting index (inclusive) of episodes to load. Defaults to 0.
            end_episode (int): The ending index (exclusive) of episodes to load.
                               If None, loads episodes from start_episode to the end. Defaults to None.
        """
        self.data_dir = data_dir
        self.frame_offset = frame_offset # Final frame offset
        self.step_size = step_size       # Step size between intermediate frames
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.num_intermediate = frame_offset // step_size
        self.episode_info = [] # Stores (task_path, episode_path, num_frames)
        self.index_map = []    # Maps dataset index to (task_path, episode_path, start_frame_idx)
        self._length = 0

        print(f"Loading video dataset from: {self.data_dir}")
        print(f"Using final frame offset: {self.frame_offset}, step size: {self.step_size}")
        print(f"Filtering episodes: start={start_episode}, end={end_episode}")
        if frame_offset % step_size != 0:
            raise ValueError(f"frame_offset ({frame_offset}) must be divisible by step_size ({step_size})")

        task_dirs = sorted([os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])

        if not task_dirs:
            print(f"Warning: No task directories found in {self.data_dir}")
            return

        for task_path in task_dirs:
            task_name = os.path.basename(task_path)
            print(f"  Processing task: {task_name}")
            # Sort numerically by episode index if possible
            episode_dirs_all = sorted([
                os.path.join(task_path, d) for d in os.listdir(task_path)
                if os.path.isdir(os.path.join(task_path, d)) and d.startswith('episode')
            ], key=lambda x: int(os.path.basename(x).split('episode')[1])) # Sort numerically by episode index

            # Filter episodes based on start_episode and end_episode
            episode_dirs = []
            for ep_path in episode_dirs_all:
                try:
                    episode_index = int(os.path.basename(ep_path).split('episode')[1])
                    is_in_range = (episode_index >= self.start_episode) and \
                                 (self.end_episode is None or episode_index < self.end_episode)
                    if is_in_range:
                        episode_dirs.append(ep_path)
                except (IndexError, ValueError):
                    print(f"    Warning: Could not parse episode index from directory name '{os.path.basename(ep_path)}'. Skipping.")
                    continue

            if not episode_dirs:
                print(f"    Warning: No episode directories found in the specified range [{self.start_episode}, {self.end_episode}) for task {task_path}")
                continue

            print(f"    Found {len(episode_dirs)} episodes in range for task '{task_name}'.")

            task_valid_frames = 0
            for episode_path in episode_dirs:
                episode_name = os.path.basename(episode_path)
                image_files = glob.glob(os.path.join(episode_path, '*.png'))
                num_frames = len(image_files)
                print(f"      Episode {episode_name} has {num_frames} frames")
                # Ensure enough frames exist for the *entire* sequence (start_frame to start_frame + frame_offset)
                if num_frames <= self.frame_offset:
                    print(f"      Skipping episode {episode_name}: only {num_frames} frames (needs > {self.frame_offset})")
                    continue

                self.episode_info.append((task_path, episode_path, num_frames))

                # Calculate valid start frames for this episode
                valid_starts = num_frames - self.frame_offset
                task_valid_frames += valid_starts

                instruction = TASK_INSTRUCTIONS.get(task_name)
                if instruction is None:
                    print(f"    Warning: No instruction found for task '{task_name}'. Skipping frames for this task.")
                    continue # Skip frames if no instruction is defined for the task

                # Populate the index map
                for start_frame_idx in range(valid_starts):
                    self.index_map.append((task_path, episode_path, start_frame_idx, instruction))

            print(f"    Found {len(episode_dirs)} episodes, {task_valid_frames} valid starting frames for task '{task_name}'.")
            self._length += task_valid_frames

        print(f"Video dataset loaded. Total valid starting frames (length): {self._length}")
        if self._length == 0:
             print("Warning: Dataset is empty. Check directory structure and frame counts.")


    def __getitem__(self, index):
        if index >= self._length:
            raise IndexError(f"Index {index} out of bounds for dataset length {self._length}")

        task_path, episode_path, start_frame_idx, instruction = self.index_map[index]

        # Define paths for input and all ground truth frames
        frame_indices = [start_frame_idx] + [start_frame_idx + (i+1) * self.step_size for i in range(self.num_intermediate)]
        frame_paths = [os.path.join(episode_path, f"{idx}.png") for idx in frame_indices]

        loaded_images = []
        try:
            for frame_path in frame_paths:
                 img = Image.open(frame_path).convert('RGB')
                 loaded_images.append(img)
        except FileNotFoundError as e:
            print(f"Error loading image for index {index}, path: {e.filename}")
            raise FileNotFoundError(f"Could not find expected image file for index {index}. Missing path: {e.filename}") from e
        except Exception as e:
             print(f"Unexpected error loading images for index {index}: {e}")
             raise RuntimeError(f"Unexpected error loading images for index {index}. Paths: {frame_paths}") from e

        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Transform all images
        transformed_images = [transform(img) for img in loaded_images]

        # Separate input image from the list of ground truth targets
        input_image = transformed_images[0]
        gt_images = transformed_images[1:] # List of tensors [i+10, i+20, ..., i+50]

        # Return input image, list of GT images, and instruction
        return input_image, gt_images, instruction

    def __len__(self):
        return self._length

class Dataset(Dataset):
    def __init__(self, data_dir, frame_offset=50, start_episode=0, end_episode=None):
        """
        Initializes the Dataset.

        Args:
            data_dir (str): Path to the main data directory containing task folders.
            frame_offset (int): Number of frames offset between input and target images.
            start_episode (int): The starting index (inclusive) of episodes to load. Defaults to 0.
            end_episode (int): The ending index (exclusive) of episodes to load.
                               If None, loads episodes from start_episode to the end. Defaults to None.
        """
        self.data_dir = data_dir
        self.frame_offset = frame_offset
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.episode_info = [] # Stores (task_path, episode_path, num_frames)
        self.index_map = []    # Maps dataset index to (task_path, episode_path, start_frame_idx)
        self._length = 0

        print(f"Loading dataset from: {self.data_dir}")
        print(f"Filtering episodes: start={start_episode}, end={end_episode}")
        task_dirs = sorted([os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])

        if not task_dirs:
            print(f"Warning: No task directories found in {self.data_dir}")
            return

        for task_path in task_dirs:
            task_name = os.path.basename(task_path)
            print(f"  Processing task: {task_name}")
            episode_dirs_all = sorted([
                os.path.join(task_path, d) for d in os.listdir(task_path)
                if os.path.isdir(os.path.join(task_path, d)) and d.startswith('episode')
            ], key=lambda x: int(os.path.basename(x).split('episode')[1])) # Sort numerically by episode index

            # Filter episodes based on start_episode and end_episode
            episode_dirs = []
            for ep_path in episode_dirs_all:
                try:
                    episode_index = int(os.path.basename(ep_path).split('episode')[1])
                    is_in_range = (episode_index >= self.start_episode) and \
                                  (self.end_episode is None or episode_index < self.end_episode)
                    if is_in_range:
                        episode_dirs.append(ep_path)
                except (IndexError, ValueError):
                    print(f"    Warning: Could not parse episode index from directory name '{os.path.basename(ep_path)}'. Skipping.")
                    continue

            if not episode_dirs:
                print(f"    Warning: No episode directories found in the specified range [{self.start_episode}, {self.end_episode}) for task {task_path}")
                continue
            
            print(f"    Found {len(episode_dirs)} episodes in range for task '{task_name}'.")

            task_valid_frames = 0
            for episode_path in episode_dirs:
                episode_name = os.path.basename(episode_path)
                # Count jpg files efficiently
                image_files = glob.glob(os.path.join(episode_path, '*.png'))
                num_frames = len(image_files)
                print(f"      Episode {episode_name} has {num_frames} frames")
                if num_frames <= self.frame_offset:
                    # print(f"      Skipping episode {episode_name}: only {num_frames} frames (needs > {self.frame_offset})")
                    continue # Skip episodes with too few frames

                self.episode_info.append((task_path, episode_path, num_frames))

                # Calculate valid start frames for this episode
                valid_starts = num_frames - self.frame_offset
                task_valid_frames += valid_starts

                # Check if task instruction exists
                instruction = TASK_INSTRUCTIONS.get(task_name)
                if instruction is None:
                    print(f"    Warning: No instruction found for task '{task_name}'. Skipping frames for this task.")
                    continue # Skip frames if no instruction is defined for the task

                # Populate the index map - store instruction here
                for start_frame_idx in range(valid_starts):
                    # Store the instruction along with path info
                    self.index_map.append((task_path, episode_path, start_frame_idx, instruction))

            print(f"    Found {len(episode_dirs)} episodes, {task_valid_frames} valid starting frames.")
            self._length += task_valid_frames

        print(f"Dataset loaded. Total valid starting frames (length): {self._length}")
        if self._length == 0:
             print("Warning: Dataset is empty. Check directory structure and frame counts.")


    def __getitem__(self, index):
        if index >= self._length:
            raise IndexError(f"Index {index} out of bounds for dataset length {self._length}")

        task_path, episode_path, start_frame_idx, instruction = self.index_map[index]

        if instruction is None:
             raise ValueError(f"Instruction not found for task '{task_name}' at index {index}. "
                              "This should not happen if initialization was correct.")

        input_frame_path = os.path.join(episode_path, f"{start_frame_idx}.png")
        output_frame_path = os.path.join(episode_path, f"{start_frame_idx + self.frame_offset}.png")

        try:
            img_x = Image.open(input_frame_path).convert('RGB')
            img_y = Image.open(output_frame_path).convert('RGB')
        except FileNotFoundError as e:
            print(f"Error loading images for index {index}: {e}")
            # Handle error appropriately, maybe return None or raise a specific exception
            # For now, re-raising the original error might be informative
            raise FileNotFoundError(f"Could not find expected image file for index {index}. "
                                    f"Input: {input_frame_path}, Output: {output_frame_path}") from e
        except Exception as e:
             print(f"Unexpected error loading images for index {index}: {e}")
             raise RuntimeError(f"Unexpected error for index {index}. "
                                 f"Input: {input_frame_path}, Output: {output_frame_path}") from e


        # Apply transformations to convert PIL images to tensors
        transform = transforms.Compose([
            transforms.Resize((256, 256)), # Example: Resize to 256x256
            transforms.ToTensor(),  # Converts PIL Image to tensor and scales values to [0.0, 1.0]
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1] 
        ])
        
        img_x = transform(img_x)
        img_y = transform(img_y)

        # Return image pair and the text instruction
        return img_x, img_y, instruction

    def __len__(self):
        return self._length
    
    
