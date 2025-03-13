import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2

import torch

from decord import VideoReader, cpu
from ssv2_dataloader import SSv2Dataset, get_transforms

import numpy as np

# Import Dataset
from torch.utils.data import Dataset, DataLoader

import os
import sys
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def test_video_loadable(video_path):
    """
    Simple test to check if a video can be opened and read
    Returns True if video can be loaded, False otherwise
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        # Try to read the first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description='Test SSv2 validation set videos')

    parser = argparse.ArgumentParser(description='Process SSv2 validation set')
    parser.add_argument('--root_dir', type=str, default = '/network/scratch/s/sonia.joseph/datasets/ssv2/somethingsomething_v2/unzipped/20bn-something-something-v2' , 
                        help='Path to SSv2 dataset root directory')
    parser.add_argument('--validation_file', type=str, default =  '/network/scratch/s/sonia.joseph/datasets/ssv2/somethingsomething_v2/unzipped/labels/validation.json',
                        help='Path to validation.json file')
    parser.add_argument('--output_file', type=str, default='valid_videos.json',
                        help='Path to save list of valid videos')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of videos to test (optional)')
    
    args = parser.parse_args()
    
    root_path = Path(args.root_dir)
    validation_path = Path(args.validation_file)
    
    # Check if validation file exists
    if not validation_path.exists():
        print(f"Error: Validation file not found: {args.validation_file}")
        sys.exit(1)
    
    # Load validation data
    with open(validation_path, 'r') as f:
        validation_data = json.load(f)
    
    print(f"Loaded {len(validation_data)} validation examples")
    
    # Apply limit if specified
    if args.limit:
        validation_data = validation_data[:args.limit]
        print(f"Limited to {args.limit} examples for testing")
    
    # Test each video
    valid_videos = []
    invalid_videos = []
    
    for example in tqdm(validation_data, desc="Testing videos"):
        video_id = example.get('id', '')
        if not video_id:
            print(f"Warning: No video ID in example")
            invalid_videos.append(example)
            continue
        
        # Check if video file exists
        video_path = root_path / f"{video_id}.webm"
        
        if not video_path.exists():
            print(f"Video not found: {video_id}")
            invalid_videos.append(example)
            continue
        
        # Test if video can be loaded
        is_valid = test_video_loadable(video_path)
        
        if is_valid:
            valid_videos.append(example)
        else:
            print(f"Video cannot be loaded: {video_id}")
            invalid_videos.append(example)
        
        # Print progress
        if (len(valid_videos) + len(invalid_videos)) % 10 == 0:
            print(f"Progress: {len(valid_videos)} valid, {len(invalid_videos)} invalid")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(valid_videos, f)
    
    # Save invalid videos
    invalid_file = Path(args.output_file).with_stem(f"{Path(args.output_file).stem}_invalid")
    with open(invalid_file, 'w') as f:
        json.dump(invalid_videos, f)
    
    print("\nTesting complete!")
    print(f"Total videos tested: {len(valid_videos) + len(invalid_videos)}")
    print(f"Valid videos: {len(valid_videos)}")
    print(f"Invalid videos: {len(invalid_videos)}")
    print(f"Valid videos saved to: {args.output_file}")
    print(f"Invalid videos saved to: {invalid_file}")



if __name__ == '__main__':
    main()