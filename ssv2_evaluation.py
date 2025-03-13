import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse
import json
from pathlib import Path

from collections import OrderedDict

from jepa.models.attentive_pooler import AttentiveClassifier

# Import your dataset and dataloader
from ssv2_dataloader import create_ssv2_dataloader, SSv2Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, dataloader, device, probe, top_k=(1, 5)):
    """
    Evaluate model accuracy on a dataloader
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for validation data
        device: Device to run evaluation on
        top_k: Tuple of k values for top-k accuracy
    
    Returns:
        dict: Dictionary with accuracy metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    batch_times = []
    
    # Store label mappings for analysis
    label_mappings = {}
    label_counts = defaultdict(int)
    label_correct = defaultdict(int)
    
    max_k = max(top_k)

    MAX = 10
    
    # Disable gradients during evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            # Record batch start time
            start_time = time.time()
            

            # switch C and F dimensions
            frames = frames.permute(0, 2, 1, 3, 4)

            # select evely spaced 16 frames in order
            B, C, F, H, W = frames.shape
            indices = torch.linspace(0, F-1, steps=16).long()
            frames = frames[:, :, indices, :, :]
            
            # Forward pass
            features = model(frames)
            outputs = probe(features)

            print("output shape:", outputs.shape)

            
            
            # Record batch processing time
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # For top-k calculation
            _, pred_indices = outputs.topk(max_k, dim=1)
            print("predictions shape:", pred_indices)
            print("labels shape:", labels)

            correct = pred_indices.eq(labels.view(-1, 1).expand_as(pred_indices))

            
            # Store predictions and labels
            all_preds.append(pred_indices.cpu())
            all_labels.append(labels.cpu())
            
            # Store label mappings if available
            if 'label_text' in batch and 'label_mapping' in batch:
                for i, label_idx in enumerate(labels.cpu().numpy()):
                    label_text = batch['label_text'][i]
                    label_mappings[label_idx.item()] = label_text
                    label_counts[label_text] += 1
                    if correct[i, 0]:  # Top-1 correct
                        label_correct[label_text] += 1

            if batch_idx > MAX:
                break
    
    # Compute top-k accuracy
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    results = {}
    for k in top_k:
        correct_k = torch.sum(
            all_preds[:, :k].eq(all_labels.view(-1, 1).expand_as(all_preds[:, :k])).any(dim=1)
        ).float()
        accuracy = correct_k / all_labels.size(0) * 100.0
        results[f'top_{k}_acc'] = accuracy.item()
    
    # Average batch processing time
    results['avg_batch_time'] = np.mean(batch_times)
    results['fps'] = len(all_labels) / np.sum(batch_times)
    
    # Per-class accuracy
    if label_counts:
        per_class_acc = {}
        for label, count in label_counts.items():
            if count > 0:
                per_class_acc[label] = (label_correct[label] / count) * 100.0
        
        # Sort by accuracy (descending)
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
        results['per_class_acc'] = dict(sorted_classes)
        
        # Best and worst performing classes
        results['best_classes'] = sorted_classes[:5]
        results['worst_classes'] = sorted_classes[-5:]
    
    return results


def load_model(model_name,  model_path):
    from vit_prisma.models.model_loader import load_hooked_model

    encoder = load_hooked_model(model_name, local_path = model_path, pretrained=False)

    # def forward_prehook(module, input):
    #     input = input[0]  # [B, C, H, W]
    #     # rearrange input

    #     print(input.shape)
    #     input = input.unsqueeze(2).repeat(1, 1,16, 1, 1)
    #     return (input)
    # encoder.register_forward_pre_hook(forward_prehook)

    encoder = encoder.to('cuda')
    
    return encoder


def load_attentive_probe(encoder, probe_path):
        # Setup probe


    checkpoint = torch.load(probe_path, map_location=DEVICE)
    checkpoint = checkpoint['classifier']


    # If the checkpoint was saved with DataParallel/DDP (has 'module.' prefix)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace('module.', '')  # remove 'module.' prefix if it exists
        new_state_dict[name] = v

    try:
        embed_dim = encoder.embed_dim
    except:
        embed_dim = encoder.cfg.d_model
    
    try:
        num_heads = encoder.num_heads
    except:
        num_heads = encoder.cfg.n_heads

    print(f"Embed dim: {embed_dim}, Num heads: {num_heads}")

    classifier = AttentiveClassifier(embed_dim= embed_dim ,num_heads = num_heads, depth=1, num_classes=174).to(DEVICE)


    classifier.load_state_dict(new_state_dict, strict=True)
    # print(f"Missing keys: {missing}", f"Unexpected keys: {unexpected}")

    # freez params
    # for param in classifier.parameters():
    #     param.requires_grad = False

    # # Put in eval mode for inference
    # classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)

    # for module in classifier.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.LayerNorm):
    #         module.eval()

    classifier.train()
    
    classifier.to(DEVICE)
    return classifier


def main():
    parser = argparse.ArgumentParser(description='Evaluate SSv2 model accuracy')
    # parser.add_argument('--model_path', type=str, required=True,
    #                     help='Path to trained model checkpoint')
    parser.add_argument('--root_dir', type=str, 
                        default= '/network/scratch/s/sonia.joseph/datasets/ssv2/somethingsomething_v2/unzipped/20bn-something-something-v2',

                        help='Path to dataset root directory')
    parser.add_argument('--annotation_file', type=str, default='valid_videos.json',
                        help='Path to validation annotation file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames per video')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--num_classes', type=int, default=174,
                        help='Number of action classes')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                        help='Path to save evaluation results')
    
    args = parser.parse_args()

    classifier_model_library = { # model_name: (model_path, probe_path)
    'vjepa_v1_vit_large_patch16': ('/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-l-16/vitl16.pth.tar', '/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-l-16/probes/ssv2-probe.pth.tar'),
    'vjepa_v1_vit_huge_patch16': ('/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-h-16/vith16.pth.tar', '/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-h-16/probes/in1k-probe.pth.tar')

}
    model_name = 'vjepa_v1_vit_large_patch16'
    model_path, probe_path = classifier_model_library[model_name]
    model = load_model(model_name, model_path)

    probe = load_attentive_probe(model, probe_path)

    
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloader for validation set
    val_loader = create_ssv2_dataloader(
        root_dir=args.root_dir,
        annotation_file=args.annotation_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        mode='val'
    )
    
    # Load model
    
    # Evaluate model
    print(f"Evaluating model on {len(val_loader.dataset)} validation examples")
    results = evaluate_model(model, val_loader, device, probe)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Top-1 Accuracy: {results['top_1_acc']:.2f}%")
    print(f"Top-5 Accuracy: {results['top_5_acc']:.2f}%")
    print(f"Average Batch Time: {results['avg_batch_time']:.4f} seconds")
    print(f"Throughput: {results['fps']:.2f} videos/second")
    
    if 'best_classes' in results:
        print("\nBest Performing Classes:")
        for class_name, accuracy in results['best_classes']:
            print(f"  {class_name}: {accuracy:.2f}%")
        
        print("\nWorst Performing Classes:")
        for class_name, accuracy in results['worst_classes']:
            print(f"  {class_name}: {accuracy:.2f}%")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()