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
    # Regular expression to match a assass name followed by a number
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
        labels = action_dict_flipped = {
    "0": "Approaching something with your camera",
    "1": "Attaching something to something",
    "2": "Bending something so that it deforms",
    "3": "Bending something until it breaks",
    "4": "Burying something in something",
    "5": "Closing something",
    "6": "Covering something with something",
    "7": "Digging something out of something",
    "8": "Dropping something behind something",
    "9": "Dropping something in front of something",
    "10": "Dropping something into something",
    "11": "Dropping something next to something",
    "12": "Dropping something onto something",
    "13": "Failing to put something into something because something does not fit",
    "14": "Folding something",
    "15": "Hitting something with something",
    "16": "Holding something",
    "17": "Holding something behind something",
    "18": "Holding something in front of something",
    "19": "Holding something next to something",
    "20": "Holding something over something",
    "21": "Laying something on the table on its side, not upright",
    "22": "Letting something roll along a flat surface",
    "23": "Letting something roll down a slanted surface",
    "24": "Letting something roll up a slanted surface, so it rolls back down",
    "25": "Lifting a surface with something on it but not enough for it to slide down",
    "26": "Lifting a surface with something on it until it starts sliding down",
    "27": "Lifting something up completely without letting it drop down",
    "28": "Lifting something up completely, then letting it drop down",
    "29": "Lifting something with something on it",
    "30": "Lifting up one end of something without letting it drop down",
    "31": "Lifting up one end of something, then letting it drop down",
    "32": "Moving away from something with your camera",
    "33": "Moving part of something",
    "34": "Moving something across a surface until it falls down",
    "35": "Moving something across a surface without it falling down",
    "36": "Moving something and something away from each other",
    "37": "Moving something and something closer to each other",
    "38": "Moving something and something so they collide with each other",
    "39": "Moving something and something so they pass each other",
    "40": "Moving something away from something",
    "41": "Moving something away from the camera",
    "42": "Moving something closer to something",
    "43": "Moving something down",
    "44": "Moving something towards the camera",
    "45": "Moving something up",
    "46": "Opening something",
    "47": "Picking something up",
    "48": "Piling something up",
    "49": "Plugging something into something",
    "50": "Plugging something into something but pulling it right out as you remove your hand",
    "51": "Poking a hole into some substance",
    "52": "Poking a hole into something soft",
    "53": "Poking a stack of something so the stack collapses",
    "54": "Poking a stack of something without the stack collapsing",
    "55": "Poking something so it slightly moves",
    "56": "Poking something so lightly that it doesn't or almost doesn't move",
    "57": "Poking something so that it falls over",
    "58": "Poking something so that it spins around",
    "59": "Pouring something into something",
    "60": "Pouring something into something until it overflows",
    "61": "Pouring something onto something",
    "62": "Pouring something out of something",
    "63": "Pretending or failing to wipe something off of something",
    "64": "Pretending or trying and failing to twist something",
    "65": "Pretending to be tearing something that is not tearable",
    "66": "Pretending to close something without actually closing it",
    "67": "Pretending to open something without actually opening it",
    "68": "Pretending to pick something up",
    "69": "Pretending to poke something",
    "70": "Pretending to pour something out of something, but something is empty",
    "71": "Pretending to put something behind something",
    "72": "Pretending to put something into something",
    "73": "Pretending to put something next to something",
    "74": "Pretending to put something on a surface",
    "75": "Pretending to put something onto something",
    "76": "Pretending to put something underneath something",
    "77": "Pretending to scoop something up with something",
    "78": "Pretending to spread air onto something",
    "79": "Pretending to sprinkle air onto something",
    "80": "Pretending to squeeze something",
    "81": "Pretending to take something from somewhere",
    "82": "Pretending to take something out of something",
    "83": "Pretending to throw something",
    "84": "Pretending to turn something upside down",
    "85": "Pulling something from behind of something",
    "86": "Pulling something from left to right",
    "87": "Pulling something from right to left",
    "88": "Pulling something onto something",
    "89": "Pulling something out of something",
    "90": "Pulling two ends of something but nothing happens",
    "91": "Pulling two ends of something so that it gets stretched",
    "92": "Pulling two ends of something so that it separates into two pieces",
    "93": "Pushing something from left to right",
    "94": "Pushing something from right to left",
    "95": "Pushing something off of something",
    "96": "Pushing something onto something",
    "97": "Pushing something so it spins",
    "98": "Pushing something so that it almost falls off but doesn't",
    "99": "Pushing something so that it falls off the table",
    "100": "Pushing something so that it slightly moves",
    "101": "Pushing something with something",
    "102": "Putting number of something onto something",
    "103": "Putting something and something on the table",
    "104": "Putting something behind something",
    "105": "Putting something in front of something",
    "106": "Putting something into something",
    "107": "Putting something next to something",
    "108": "Putting something on a flat surface without letting it roll",
    "109": "Putting something on a surface",
    "110": "Putting something on the edge of something so it is not supported and falls down",
    "111": "Putting something onto a slanted surface but it doesn't glide down",
    "112": "Putting something onto something",
    "113": "Putting something onto something else that cannot support it so it falls down",
    "114": "Putting something similar to other things that are already on the table",
    "115": "Putting something that can't roll onto a slanted surface, so it slides down",
    "116": "Putting something that can't roll onto a slanted surface, so it stays where it is",
    "117": "Putting something that cannot actually stand upright upright on the table, so it falls on its side",
    "118": "Putting something underneath something",
    "119": "Putting something upright on the table",
    "120": "Putting something, something and something on the table",
    "121": "Removing something, revealing something behind",
    "122": "Rolling something on a flat surface",
    "123": "Scooping something up with something",
    "124": "Showing a photo of something to the camera",
    "125": "Showing something behind something",
    "126": "Showing something next to something",
    "127": "Showing something on top of something",
    "128": "Showing something to the camera",
    "129": "Showing that something is empty",
    "130": "Showing that something is inside something",
    "131": "Something being deflected from something",
    "132": "Something colliding with something and both are being deflected",
    "133": "Something colliding with something and both come to a halt",
    "134": "Something falling like a feather or paper",
    "135": "Something falling like a rock",
    "136": "Spilling something behind something",
    "137": "Spilling something next to something",
    "138": "Spilling something onto something",
    "139": "Spinning something so it continues spinning",
    "140": "Spinning something that quickly stops spinning",
    "141": "Spreading something onto something",
    "142": "Sprinkling something onto something",
    "143": "Squeezing something",
    "144": "Stacking number of something",
    "145": "Stuffing something into something",
    "146": "Taking one of many similar things on the table",
    "147": "Taking something from somewhere",
    "148": "Taking something out of something",
    "149": "Tearing something into two pieces",
    "150": "Tearing something just a little bit",
    "151": "Throwing something",
    "152": "Throwing something against something",
    "153": "Throwing something in the air and catching it",
    "154": "Throwing something in the air and letting it fall",
    "155": "Throwing something onto a surface",
    "156": "Tilting something with something on it slightly so it doesn't fall down",
    "157": "Tilting something with something on it until it falls off",
    "158": "Tipping something over",
    "159": "Tipping something with something in it over, so something in it falls out",
    "160": "Touching (without moving) part of something",
    "161": "Trying but failing to attach something to something because it doesn't stick",
    "162": "Trying to bend something unbendable so nothing happens",
    "163": "Trying to pour something into something, but missing so it spills next to it",
    "164": "Turning something upside down",
    "165": "Turning the camera downwards while filming something",
    "166": "Turning the camera left while filming something",
    "167": "Turning the camera right while filming something",
    "168": "Turning the camera upwards while filming something",
    "169": "Twisting (wringing) something wet until water comes out",
    "170": "Twisting something",
    "171": "Uncovering something",
    "172": "Unfolding something",
    "173": "Wiping something off of something"


        }
        return labels
    
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
    # val_annotation = '/network/scratch/s/sonia.joseph/datasets/ssv2/somethingsomething_v2/unzipped/labels/validation.json'
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