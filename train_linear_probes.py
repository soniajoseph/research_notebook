import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch import einsum

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LAYERS = 12 # Based on the provided code
NUM_CLASSES = 1000  # ImageNet classes
BATCH_SIZE = 64
NUM_EPOCHS = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = "probe_checkpoints"  # Directory to save probe checkpoints


EVAL_BATCHES = 1 # Number of batches to use for evaluation
TRAIN_MAX = 1

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value"""
        self.val = float(val)  # Ensure val is a float
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy for the top-k predictions and returns the predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        raw_max_logits_values = output.max(dim=1).values
    
        # Get top-k predictions
        topk_preds = output.topk(maxk, dim=1, largest=True, sorted=True).indices

        # Compute top-1 accuracy
        top1_acc = 100.0 * topk_preds[:, 0].eq(target).sum().float() / batch_size

        # Compute top-k accuracy
        correct = topk_preds.eq(target.view(-1, 1))  # Shape: (batch_size, maxk)
        accs = [100.0 * correct[:, :k].float().sum() / batch_size for k in topk]

        return accs, topk_preds[:, 0], topk_preds[:, :5], raw_max_logits_values

def average_logit_value_across_all_classes(residual_stack, cache, encoder, mean=False):
    """Compute logit values across all classes for a residual stack."""
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=0
    )

    all_residual_directions = encoder.tokens_to_residual_directions(np.arange(1000))  # Get all residual directions

    logit_predictions = einsum(
        "layer batch d_model, class d_model -> batch layer class",
        scaled_residual_stack[1:],
        all_residual_directions,
    )
    
    if mean:
        logit_predictions = logit_predictions.mean(axis=0)

    return logit_predictions

class LinearProbe(nn.Module):
    """Linear probe for a specific layer."""
    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def train_layer_probe(encoder, train_loader, val_loader, layer_idx, feature_dim, num_epochs=NUM_EPOCHS):
    """Train a linear probe for a specific layer."""
    print(f"Training probe for layer {layer_idx}...")
    
    # Create probe
    probe = LinearProbe(feature_dim, NUM_CLASSES).to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"probe_layer_{layer_idx}.pt")
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint for layer {layer_idx}...")
        probe.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        return probe
    
    # Training loop
    best_val_acc = 0
    best_probe_state = None
    
    count = 0
    for epoch in range(num_epochs):
        # Training
        probe.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Extract features for this batch only
            with torch.no_grad():
                _, cache = encoder.run_with_cache(images)
                features = cache[f'blocks.{layer_idx}.hook_resid_post'][:, 0]  # Class token
            
            # Forward pass
            outputs = probe(features)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            batch_acc = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
            
            # Update metrics
            train_loss.update(loss.item(), images.size(0))
            train_acc.update(batch_acc, images.size(0))

            count += 1
            if count > TRAIN_MAX:
                break
        
        # Validation
        val_acc = evaluate_layer_probe(encoder, val_loader, probe, layer_idx)
        
        print(f"Layer {layer_idx}, Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_probe_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            print(f"New best for layer {layer_idx}: {val_acc:.2f}%")
    
    # Load best model
    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)
    
    # Save checkpoint
    torch.save(probe.state_dict(), checkpoint_path)
    
    print(f"Best validation accuracy for layer {layer_idx}: {best_val_acc:.2f}%")
    
    return probe

def evaluate_layer_probe(encoder, val_loader, probe, layer_idx, max_batches=EVAL_BATCHES):
    """Evaluate a single probe on validation data."""
    probe.eval()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc=f"Validating layer {layer_idx}")):
            if batch_idx >= max_batches:
                break
                
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Extract features
            _, cache = encoder.run_with_cache(images)
            features = cache[f'blocks.{layer_idx}.hook_resid_post'][:, 0]  # Class token
            
            # Get predictions
            outputs = probe(features)
            
            # Calculate accuracy
            (acc1, acc5), _, _, _ = accuracy(outputs, targets, topk=(1, 5))
            
            # Update meters
            top1_meter.update(acc1.item(), images.size(0))
            top5_meter.update(acc5.item(), images.size(0))
    
    return top1_meter.avg

def train_linear_probes_efficient(encoder, train_loader, val_loader, feature_dim):
    """Train linear probes for each layer of the encoder in a memory-efficient way."""
    probes = []
    layer_results = []
    
    for layer_idx in range(NUM_LAYERS):
        # Train probe for current layer
        probe = train_layer_probe(encoder, train_loader, val_loader, layer_idx, feature_dim)
        probes.append(probe)
        
        # Evaluate probe on validation set
        top1_acc = evaluate_layer_probe(encoder, val_loader, probe, layer_idx)
        
        layer_results.append((top1_acc))
        
        print(f"Layer {layer_idx} final validation: Top-1 = {top1_acc:.2f}%")
    
    return probes, layer_results

def evaluate_imagenet_probes(encoder, val_loader, probes, max_batches=EVAL_BATCHES):
    """Evaluate the trained linear probes on the validation set."""
    # Create average meters for each layer
    top1_meters = [AverageMeter() for _ in range(NUM_LAYERS)]
    top5_meters = [AverageMeter() for _ in range(NUM_LAYERS)]
    
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(tqdm(val_loader, desc="Evaluating Probes")):
            if batch_idx >= max_batches:
                break
                
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                images = images.to(DEVICE)
                target = target.to(DEVICE)
                
                # Get representations
                _, cache = encoder.run_with_cache(images)
                
                # Evaluate each layer's probe
                for layer_idx in range(NUM_LAYERS):
                    # Extract features
                    features = cache[f'blocks.{layer_idx}.hook_resid_post'][:, 0]  # Class token
                    
                    # Get probe predictions
                    logits = probes[layer_idx](features)
                    
                    # Calculate accuracy
                    (acc1, acc5), _, _, _ = accuracy(logits, target, topk=(1, 5))
                    
                    # Update meters
                    top1_meters[layer_idx].update(acc1.item(), images.size(0))
                    top5_meters[layer_idx].update(acc5.item(), images.size(0))
    
    # Return results
    return [(meter.avg, top5_meters[i].avg) for i, meter in enumerate(top1_meters)]

def evaluate_imagenet_probe_original(encoder, val_loader, max_batches=EVAL_BATCHES):
    """Original evaluation function that uses residual directions."""
    # Create average meters for each layer
    top1_meters = [AverageMeter() for _ in range(NUM_LAYERS)]
    top5_meters = [AverageMeter() for _ in range(NUM_LAYERS)]
    
    raw_logits_per_layer = [[] for _ in range(NUM_LAYERS)]
    
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(tqdm(val_loader, desc="Evaluating Original Method")):
            if batch_idx >= max_batches:
                break

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                images = images.to(DEVICE)
                target = target.to(DEVICE)

                # Get representations
                _, cache = encoder.run_with_cache(images)

                accumulated_residual, layer_labels = cache.accumulated_resid(layer=-1, incl_mid=False, pos_slice=0, return_labels=True)
                
                all_layers_logits = average_logit_value_across_all_classes(
                    accumulated_residual, cache, encoder, mean=False
                )

                for layer_idx in range(all_layers_logits.shape[1]):
                    # Get logits for current layer
                    layer_logits = all_layers_logits[:, layer_idx, :]
                    
                    # Calculate accuracy metrics
                    (acc1, acc5), top1_preds, top5_preds, raw_max_logits = accuracy(
                        layer_logits, target, topk=(1, 5)
                    )
                    
                    # Update metrics for this layer
                    top1_meters[layer_idx].update(acc1, images.size(0))
                    top5_meters[layer_idx].update(acc5, images.size(0))
                    raw_logits_per_layer[layer_idx].extend(raw_max_logits.cpu().numpy().tolist())
    
    return top1_meters, top5_meters, raw_logits_per_layer

def plot_accuracy_comparison(linear_probe_results, original_results=None, title="Layer-wise Accuracy Comparison"):
    """Plot accuracy comparison between linear probes and original method."""
    layers = list(range(NUM_LAYERS))
    
    # Extract accuracies
    linear_top1 = [res[0] for res in linear_probe_results]
    linear_top5 = [res[1] for res in linear_probe_results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot Top-1 accuracies
    plt.subplot(1, 2, 1)
    plt.plot(layers, linear_top1, 'b-', marker='o', label='Linear Probe')
    if original_results:
        original_top1 = [meter.avg for meter in original_results[0]]
        plt.plot(layers, original_top1, 'r-', marker='x', label='Original Method')
    plt.xlabel('Layer')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Top-1 Accuracy by Layer')
    plt.legend()
    plt.grid(True)
    
    # Plot Top-5 accuracies
    plt.subplot(1, 2, 2)
    plt.plot(layers, linear_top5, 'b-', marker='o', label='Linear Probe')
    if original_results:
        original_top5 = [meter.avg for meter in original_results[1]]
        plt.plot(layers, original_top5, 'r-', marker='x', label='Original Method')
    plt.xlabel('Layer')
    plt.ylabel('Top-5 Accuracy (%)')
    plt.title('Top-5 Accuracy by Layer')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()

def main():
    from vit_prisma_legacy.models.base_vit import HookedViT

    encoder = HookedViT.from_pretrained("vit_base_patch32_224",
                                         center_writing_weights=True,
                                         center_unembed=True,
                                         fold_ln=True,
                                         refactor_factored_attn_matrices=True,
                                        )
    encoder = encoder.to(DEVICE)
    encoder.eval()  # Set model to evaluation mode

    # Data loading
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_data_path = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
    val_data_path = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"

    train_dataset = datasets.ImageFolder(
        root=train_data_path,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    val_dataset = datasets.ImageFolder(
        root=val_data_path,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    # Use more workers to speed up data loading
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=8, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=8, pin_memory=True
    )
    
    # Define the dimension of features (ViT base model dimension)
    feature_dim = 768
    
    # Train linear probes using memory-efficient approach
    print("Training linear probes...")
    probes, _ = train_linear_probes_efficient(encoder, train_loader, val_loader, feature_dim)
    
    # Final evaluation
    print("Final evaluation of linear probes...")
    linear_probe_results = evaluate_imagenet_probes(encoder, val_loader, probes)
    
    # Optional: evaluate using original method
    run_original_method = False  # Set to False to skip this
    original_results = None
    if run_original_method:
        print("Evaluating using original method...")
        original_results = evaluate_imagenet_probe_original(encoder, val_loader)
    
    # Plot comparison
    plot_accuracy_comparison(linear_probe_results, original_results)
    
    # Print final results
    print("\n===== Linear Probe Results =====")
    for layer_idx, (top1, top5) in enumerate(linear_probe_results):
        print(f"Layer {layer_idx}: Top-1 = {top1:.2f}%, Top-5 = {top5:.2f}%")
    
    if original_results:
        print("\n===== Original Method Results =====")
        for layer_idx, meter in enumerate(original_results[0]):
            print(f"Layer {layer_idx}: Top-1 = {meter.avg:.2f}%, Top-5 = {original_results[1][layer_idx].avg:.2f}%")

if __name__ == "__main__":
    main()