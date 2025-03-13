import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import glob
import cv2
from decord import VideoReader, cpu
import pandas as pd

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import glob
import cv2
from decord import VideoReader, cpu
import pandas as pd

import re

def parse_class_labels(text_content):
    """
    Parse the class labels and their counts from the provided text content.
    
    Args:
        text_content (str): The text containing class labels and their counts
        
    Returns:
        dict: A dictionary mapping class index to class name
        dict: A dictionary mapping class name to count
    """
    # Regular expression to match a class name followed by a number
    pattern = r'(.*?)\s+(\d+)$'
    
    lines = text_content.strip().split('\n')
    
    class_dict = {}
    count_dict = {}
    
    class_idx = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            class_name = match.group(1).strip()
            count = int(match.group(2))
            
            class_dict[class_idx] = class_name
            count_dict[class_name] = count
            
            class_idx += 1
    
    return class_dict, count_dict


def extract_class_mapping():
    """
    Extract the class mapping from the dataset instructions.
    
    This function contains the hardcoded text from the dataset instructions.
    It's better to load from a file, but this is a fallback.
    
    Returns:
        dict: A dictionary mapping class index to class name
    """
    # Text containing the class names and counts
    class_text = """Unfolding something 1266
Pushing something from left to right 3442
Attaching something to something 1227
Something falling like a rock 2079
Something falling like a feather or paper 1858
Bending something so that it deforms 798
Pretending to poke something 754
Pulling something from left to right 1908
Pretending to put something onto something 740
Scooping something up with something 1123
Putting something next to something 2431
Poking a stack of something so the stack collapses 367
Showing something next to something 1185
Poking something so that it falls over 892
Pouring something onto something 403
Putting something and something on the table 1353
Bending something until it breaks 718
Putting something underneath something 748
Squeezing something 2631
Trying to bend something unbendable so nothing happens 991
Digging something out of something 522
Moving part of something 905
Moving something towards the camera 994
Dropping something behind something 991
Holding something in front of something 2203
Something colliding with something and both come to a halt 547
Spilling something onto something 474
Pretending to put something on a surface 1644
Pretending to throw something 1019
Sprinkling something onto something 540
Tilting something with something on it until it falls off 1272
Pulling something onto something 343
Pulling something from right to left 1886
Pulling two ends of something but nothing happens 643
Turning the camera upwards while filming something 1021
Pretending to sprinkle air onto something 543
Poking something so lightly that it doesn't or almost doesn't move 2430
Poking a hole into something soft 258
Pretending to take something out of something 1045
Moving something and something closer to each other 2298
Putting something that can't roll onto a slanted surface, so it slides down 442
Putting something that cannot actually stand upright upright on the table, so it falls on its side 837
Moving something and something so they pass each other 582
Moving something up 3750
Moving something across a surface without it falling down 832
Moving something down 3242
Pulling two ends of something so that it gets stretched 438
Letting something roll down a slanted surface 876
Pretending to put something behind something 746
Pushing something so that it slightly moves 2418
Dropping something onto something 1623
Throwing something in the air and catching it 1177
Putting something onto a slanted surface but it doesn't glide down 183
Putting something on a flat surface without letting it roll 553
Moving away from something with your camera 1199
Putting something in front of something 1094
Pushing something from right to left 3195
Turning the camera left while filming something 1239
Putting something onto something else that cannot support it so it falls down 442
Pushing something so it spins 845
Tipping something with something in it over, so something in it falls out 447
Showing something behind something 2315
Putting something on a surface 4081
Pulling something from behind of something 586
Touching (without moving) part of something 1763
Laying something on the table on its side, not upright 950
Putting number of something onto something 1180
Turning the camera downwards while filming something 976
Taking something from somewhere 1290
Lifting something up completely, then letting it drop down 1851
Pretending to scoop something up with something 389
Dropping something next to something 1232
Twisting something 1131
Throwing something onto a surface 1035
Pretending to turn something upside down 888
Holding something behind something 1374
Moving something closer to something 1426
Moving something and something so they collide with each other 577
Putting something into something 2783
Putting something upright on the table 980
Dropping something in front of something 1131
Picking something up 1456
Pouring something into something until it overflows 352
Pushing something onto something 419
Pretending to open something without actually opening it 1911
Pretending to put something underneath something 373
Pulling something out of something 736
Showing a photo of something to the camera 916
Holding something next to something 1893
Putting something behind something 1428
Lifting a surface with something on it until it starts sliding down 405
Plugging something into something but pulling it right out as you remove your hand 1176
Moving something across a surface until it falls down 883
Failing to put something into something because something does not fit 353
Poking something so that it spins around 185
Pouring something into something 1530
Showing that something is empty 2209
Lifting up one end of something without letting it drop down 1613
Pretending to squeeze something 856
Throwing something 2626
Pushing something so that it falls off the table 2240
Pretending or trying and failing to twist something 404
Pretending to put something next to something 1297
Wiping something off of something 873
Pushing something off of something 687
Pushing something with something 1804
Letting something roll along a flat surface 1163
Pretending to pick something up 1969
Pretending or failing to wipe something off of something 490
Showing that something is inside something 1547
Removing something, revealing something behind 1069
Letting something roll up a slanted surface, so it rolls back down 441
Moving something and something away from each other 2062
Putting something onto something 1850
Pretending to close something without actually closing it 1122
Tearing something just a little bit 2025
Showing something on top of something 1301
Something being deflected from something 492
Covering something with something 3530
Moving something away from the camera 986
Trying but failing to attach something to something because it doesn't stick 660
Putting something on the edge of something so it is not supported and falls down 638
Spilling something behind something 143
Holding something over something 1804
Twisting (wringing) something wet until water comes out 408
Taking something out of something 2259
Piling something up 1145
Opening something 1869
Trying to pour something into something, but missing so it spills next to it 265
Lifting up one end of something, then letting it drop down 1850
Spinning something so it continues spinning 1168
Spreading something onto something 535
Poking a hole into some substance 115
Stacking number of something 1463
Poking a stack of something without the stack collapsing 276
Pretending to spread air onto something 225
Uncovering something 3004
Spinning something that quickly stops spinning 1587
Pretending to be tearing something that is not tearable 1256
Putting something similar to other things that are already on the table 2339
Rolling something on a flat surface 1773
Pretending to take something from somewhere 1437
Taking one of many similar things on the table 2969
Holding something 1851
Folding something 1542
Throwing something against something 1475
Throwing something in the air and letting it fall 1038
Pretending to put something into something 1165
Showing something to the camera 1061
Something colliding with something and both are being deflected 653
Putting something, something and something on the table 1211
Plugging something into something 2252
Lifting something up completely without letting it drop down 1906
Moving something away from something 1352
Tipping something over 896
Pretending to pour something out of something, but something is empty 445
Turning the camera right while filming something 1239
Turning something upside down 2943
Pulling two ends of something so that it separates into two pieces 313
Tilting something with something on it slightly so it doesn't fall down 829
Hitting something with something 2234
Lifting a surface with something on it but not enough for it to slide down 268
Approaching something with your camera 1349
Closing something 1482
Lifting something with something on it 2016
Putting something that can't roll onto a slanted surface, so it stays where it is 447
Spilling something next to something 240
Pouring something out of something 514
Poking something so it slightly moves 1599
Tearing something into two pieces 2849
Pushing something so that it almost falls off but doesn't 1321
Burying something in something 687
Stuffing something into something 1998
Dropping something into something 1222"""

    class_dict, count_dict = parse_class_labels(class_text)
    return class_dict, count_dict
    
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import glob
import cv2
from decord import VideoReader, cpu, gpu
import pandas as pd
import re
import concurrent.futures
from tqdm import tqdm

# Configure Decord globally
import decord
decord.bridge.set_bridge('torch')  # Use PyTorch tensors

def parse_class_labels(text_content):
    """
    Parse the class labels and their counts from the provided text content.
    
    Args:
        text_content (str): The text containing class labels and their counts
        
    Returns:
        dict: A dictionary mapping class index to class name
        dict: A dictionary mapping class name to count
    """
    # Regular expression to match a class name followed by a number
    pattern = r'(.*?)\s+(\d+)$'
    
    lines = text_content.strip().split('\n')
    
    class_dict = {}
    count_dict = {}
    
    class_idx = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            class_name = match.group(1).strip()
            count = int(match.group(2))
            
            class_dict[class_idx] = class_name
            count_dict[class_name] = count
            
            class_idx += 1
    
    return class_dict, count_dict


def extract_class_mapping():
    """
    Extract the class mapping from the dataset instructions.
    
    This function contains the hardcoded text from the dataset instructions.
    It's better to load from a file, but this is a fallback.
    
    Returns:
        dict: A dictionary mapping class index to class name
    """
    # Text containing the class names and counts (truncated for brevity)
    class_text = """Unfolding something 1266
Pushing something from left to right 3442
Attaching something to something 1227
Something falling like a rock 2079
Something falling like a feather or paper 1858
"""  # Truncated for brevity
    class_dict, count_dict = parse_class_labels(class_text)
    return class_dict, count_dict
    

class SSv2Dataset(Dataset):
    """
    Dataloader for the Something-Something-V2 dataset with optimized CUDA support
    """
    def __init__(self, root_dir, annotation_file, transform=None, num_frames=32, mode='train', 
                 use_cuda=False, cache_size=200, num_retries=3):
        """
        Args:
            root_dir (string): Directory with all the videos.
            annotation_file (string): Path to the annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_frames (int): Number of frames to sample from the video.
            mode (string): 'train', 'val', or 'test' mode.
            use_cuda (bool): Whether to use CUDA for video decoding if available.
            cache_size (int): Number of videos to cache in memory.
            num_retries (int): Number of retries for loading a video before giving up.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_frames = num_frames
        self.mode = mode
        
        # Check if CUDA is actually available for Decord
        try:
            # Test if GPU context works
            test_ctx = gpu(0)
            self.use_cuda = use_cuda and torch.cuda.is_available()
            self.ctx = gpu(0) if self.use_cuda else cpu(0)
            print("CUDA is available for Decord")
        except Exception as e:
            print(f"CUDA not available for Decord: {e}")
            self.use_cuda = False
            self.ctx = cpu(0)
        
        self.cache_size = cache_size
        self.cache = {}  # Video cache
        self.num_retries = num_retries
        
        # Load annotations
        if annotation_file.endswith('.json'):
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        elif annotation_file.endswith('.csv'):
            # If using CSV format
            self.annotations = pd.read_csv(annotation_file).to_dict('records')
            
        # Load class labels
        self.classes = self._load_class_labels()
        
        # Keep track of problematic videos to avoid repeated errors
        self.failed_videos = set()
        
        # Pre-find all video paths
        self._prefind_video_paths()
        
    def _prefind_video_paths(self):
        """
        Pre-find paths for all videos to avoid repeated searches.
        """
        self.video_paths = {}
        
        # First check if the root dir has 20bn-something-something-v2 subdirectory
        ssv2_subdir = self.root_dir / '20bn-something-something-v2'
        if ssv2_subdir.exists():
            self.main_dir = ssv2_subdir
        else:
            self.main_dir = self.root_dir
        
        # Check if we need to search in numbered directories
        has_numbered_dirs = False
        for i in range(1):  # Just check the first one
            if (self.root_dir / f'20bn-something-something-v2-{i:02d}').exists():
                has_numbered_dirs = True
                break
        
        self.has_numbered_dirs = has_numbered_dirs
        
        # Determine search strategy based on directory structure
        print(f"Using main directory: {self.main_dir}")
        print(f"Using numbered directories: {has_numbered_dirs}")
        
    def _load_class_labels(self):
        """Load class labels for the dataset."""
        try:
            # Try to load classes using the parser
            class_dict, _ = extract_class_mapping()
            return class_dict
        except Exception as e:
            print(f"Warning: Could not load class labels: {e}")
            # Fallback to a minimal set of classes
            return {
                0: "Unfolding something",
                1: "Pushing something from left to right",
                # Add more as needed
            }
    
    def __len__(self):
        return len(self.annotations)
    
    def _get_video_path(self, video_id):
        """
        Find the video file path based on its ID.
        """
        # Convert to string and ensure it's a proper file ID
        video_id = str(video_id)
        
        # Check if path is already cached
        if video_id in self.video_paths:
            return self.video_paths[video_id]
        
        # Try direct path in main directory first (most common case)
        video_path = self.main_dir / f'{video_id}.webm'
        if video_path.exists():
            self.video_paths[video_id] = str(video_path)
            return str(video_path)
        
        # Try searching in numbered directories if they exist
        if self.has_numbered_dirs:
            for i in range(14):  # Parts 00-13
                part_dir = self.root_dir / f'20bn-something-something-v2-{i:02d}'
                if part_dir.exists():
                    video_path = part_dir / f'{video_id}.webm'
                    if video_path.exists():
                        self.video_paths[video_id] = str(video_path)
                        return str(video_path)
        
        # Try other extensions as fallback
        for ext in ['.mp4', '.avi', '.mov']:
            video_path = self.main_dir / f'{video_id}{ext}'
            if video_path.exists():
                self.video_paths[video_id] = str(video_path)
                return str(video_path)
        
        raise FileNotFoundError(f"Video ID {video_id} not found.")
    
    def _load_video_decord(self, video_path):
        """
        Load video frames using Decord with proper error handling.
        """
        for retry in range(self.num_retries):
            try:
                # Always try CPU first since CUDA is failing
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
                
                # Calculate frame indices to sample
                total_frames = len(vr)
                if total_frames <= 0:
                    raise ValueError(f"Video has no frames: {video_path}")
                
                if total_frames <= self.num_frames:
                    indices = torch.arange(total_frames)
                    # If video is too short, repeat the last frame
                    if total_frames < self.num_frames:
                        indices = torch.cat([indices, indices[-1].repeat(self.num_frames - total_frames)])
                else:
                    # Uniformly sample frames
                    indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
                
                # Load the sampled frames
                frames = vr.get_batch(indices)
                
                # If frames are numpy array, convert to torch
                if not isinstance(frames, torch.Tensor):
                    frames = torch.from_numpy(frames)
                
                # Convert to [T, C, H, W] format
                if frames.shape[-1] == 3:  # Last dim is channels, so permute
                    frames = frames.permute(0, 3, 1, 2)
                
                # Normalize to [0, 1]
                frames = frames.float() / 255.0
                
                return frames
                
            except Exception as e:
                if retry < self.num_retries - 1:
                    print(f"Retry {retry+1}/{self.num_retries} for {video_path}: {str(e)}")
                    # Reduce number of threads for next attempt
                    num_threads = max(1, 2 - retry)
                else:
                    print(f"All Decord retries failed for {video_path}: {str(e)}")
        
        # If all retries fail, fall back to OpenCV
        return self._load_video_opencv(video_path)
    
    def _load_video_opencv(self, video_path):
        """
        Fallback method to load video frames using OpenCV.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            # Determine which frames to sample
            if total_frames <= self.num_frames:
                indices = np.arange(total_frames)
                # Pad with last frame if video is too short
                if total_frames < self.num_frames:
                    indices = np.pad(indices, (0, self.num_frames - total_frames), 'edge')
            else:
                # Uniformly sample frames
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            # Read selected frames
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    # Use black frame as fallback
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor [C, H, W]
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
            
            cap.release()
            
            # Stack frames to tensor [T, C, H, W]
            return torch.stack(frames)
            
        except Exception as e:
            print(f"OpenCV fallback also failed for {video_path}: {str(e)}")
            # Return a dummy tensor
            return torch.zeros((self.num_frames, 3, 240, 320))
    
    def _load_video(self, video_path):
        """
        Load video frames with caching and multiple fallback methods.
        """
        video_id = Path(video_path).stem
        
        # Check if video is known to fail
        if video_id in self.failed_videos:
            return torch.zeros((self.num_frames, 3, 240, 320))
        
        # Check if video is in cache
        if video_id in self.cache:
            return self.cache[video_id]
        
        try:
            # Try to load with Decord first (faster, especially with CUDA)
            frames = self._load_video_decord(video_path)
            
            # Cache the result
            if len(self.cache) >= self.cache_size:
                # Remove oldest item
                self.cache.pop(next(iter(self.cache)))
            self.cache[video_id] = frames
            
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Mark as failed to avoid retrying
            self.failed_videos.add(video_id)
            # Return a dummy tensor
            return torch.zeros((self.num_frames, 3, 240, 320))
        

    # Simple usage:
    # Replace the dataloader's collate_fn with this function

    # Modified getitem function to handle string labels
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get annotation for the video
        annotation = self.annotations[idx]
        
        # Get video ID
        video_id = annotation.get('id', str(idx))
        
        # Get label - handles both numeric and string labels
        if 'label_id' in annotation and isinstance(annotation['label_id'], int):
            # Use numeric label_id if available
            label = annotation['label_id']
        elif 'label' in annotation:
            # Use string label if available
            label = annotation['label']
        elif 'template' in annotation:
            # Use template as fallback
            label = annotation['template']
        else:
            # No label information
            label = -1
        
        # Get video path
        try:
            video_path = self._get_video_path(video_id)
            frames = self._load_video(video_path)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            frames = torch.zeros((self.num_frames, 3, 224, 224))
        
        # Apply transformations
        if self.transform:
            transformed_frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            transformed_frames = frames
        
        sample = {
            'frames': transformed_frames,
            'label': label,
            'video_id': video_id
        }
        
        return sample


def get_transforms(mode='train'):
    """
    Get video transforms for training or validation.
    """
    if mode == 'train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def ssv2_collate_fn(batch):
    """
    Custom collate function that handles string labels and missing/corrupt videos.
    Returns a valid batch even if some videos failed to load.
    """
    # Filter out None values and frames with all zeros (failed videos)
    valid_batch = []
    for item in batch:
        if item is not None and isinstance(item, dict) and 'frames' in item:
            # Check if this is a dummy tensor (all zeros)
            if not torch.all(item['frames'] == 0):
                valid_batch.append(item)
    
    # If no valid items, return a dummy batch
    if len(valid_batch) == 0:
        return {
            'frames': torch.zeros((1, 32, 3, 224, 224)),
            'label': torch.tensor([-1]),
            'video_id': ['dummy']
        }
    
    # Stack the valid items
    frames = torch.stack([item['frames'] for item in valid_batch])
    
    # Handle string labels by using indices instead
    # Convert all labels to strings in case some are numeric
    labels = [str(item['label']) for item in valid_batch]
    
    # Create a mapping from string labels to numeric IDs for this batch
    label_to_id = {label: i for i, label in enumerate(sorted(set(labels)))}
    numeric_labels = torch.tensor([label_to_id[label] for label in labels])
    
    video_ids = [item['video_id'] for item in valid_batch]
    
    return {
        'frames': frames,
        'label': numeric_labels,
        'label_text': labels,  # Keep original text labels
        'video_id': video_ids,
        'label_mapping': label_to_id  # Include the mapping for reference
    }


def create_ssv2_dataloader(root_dir, annotation_file, batch_size=16, num_workers=4, 
                           num_frames=32, mode='train', use_cuda=False, prefetch_factor=2):
    """
    Create a DataLoader for the SSv2 dataset with optimized settings.
    
    Args:
        root_dir (string): Directory with all the videos.
        annotation_file (string): Path to the annotation file.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of worker processes for data loading.
        num_frames (int): Number of frames to sample from each video.
        mode (string): 'train', 'val', or 'test' mode.
        use_cuda (bool): Whether to use CUDA for video decoding if available.
        prefetch_factor (int): Number of batches to prefetch per worker.
        
    Returns:
        DataLoader: PyTorch DataLoader for the SSv2 dataset.
    """
    transform = get_transforms(mode)
    
    # Create dataset
    dataset = SSv2Dataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        transform=transform,
        num_frames=num_frames,
        mode=mode,
        use_cuda=use_cuda
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
        collate_fn=ssv2_collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    return dataloader




# Example usage
if __name__ == '__main__':
    # Paths
    root_dir = '/network/scratch/s/sonia.joseph/datasets/ssv2/somethingsomething_v2/unzipped/20bn-something-something-v2'
    train_annotation = '/network/scratch/s/sonia.joseph/datasets/ssv2/somethingsomething_v2/unzipped/labels/train.json'
    val_annotation = '/network/scratch/s/sonia.joseph/datasets/ssv2/somethingsomething_v2/unzipped/labels/validation.json'
    val_annotation = 'valid_videos.json'
    
    # # Create dataloaders
    # train_loader = create_ssv2_dataloader(
    #     root_dir=root_dir,
    #     annotation_file=train_annotation,
    #     batch_size=1,
    #     num_workers=4,
    #     num_frames=32,
    #     mode='train'
    # )
    
    val_loader = create_ssv2_dataloader(
        root_dir=root_dir,
        annotation_file=val_annotation,
        batch_size=16,
        num_workers=4,
        num_frames=32,
        mode='val'
    )
    
    # Test the dataloader
    for batch in val_loader:
        frames = batch['frames']
        labels = batch['label']
        video_ids = batch['video_id']
        
        print(f"Batch shape: {frames.shape}")  # Should be [B, T, C, H, W]
        print(f"Labels: {labels}")
        print(f"Video IDs: {video_ids}")
        break