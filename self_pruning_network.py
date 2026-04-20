"""
Self-Pruning Neural Network for CIFAR-10 Classification
Author: AI Engineering Internship Case Study
Description: Implementation of a neural network that learns to prune itself during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


class PrunableLinear(nn.Module):
    """
    Custom Linear layer with learnable gates for self-pruning.
    
    Each weight has an associated gate (0 to 1) that controls its contribution.
    Gates near 0 effectively prune the weight.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Gate scores - learnable parameters for pruning
        # These will be transformed to [0, 1] range using sigmoid
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and gates with appropriate values"""
        # Initialize weights using Kaiming uniform (good for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize gate scores to positive values (so gates start near 1 after sigmoid)
        # This ensures the network starts with all connections active
        nn.init.constant_(self.gate_scores, 2.0)
    
    def forward(self, x):
        """
        Forward pass with gated weights.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Transform gate_scores to [0, 1] range using sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gates to weights (element-wise multiplication)
        pruned_weights = self.weight * gates
        
        # Perform standard linear transformation
        # output = input @ pruned_weights.T + bias
        output = torch.nn.functional.linear(x, pruned_weights, self.bias)
        
        return output
    
    def get_gates(self):
        """Return the current gate values (after sigmoid)"""
        return torch.sigmoid(self.gate_scores)


class SelfPruningNetwork(nn.Module):
    """
    Neural Network for CIFAR-10 classification with self-pruning capability.
    
    Architecture: Conv layers for feature extraction + Prunable FC layers
    """
    
    def __init__(self):
        super(SelfPruningNetwork, self).__init__()
        
        # Convolutional layers for feature extraction
        # CIFAR-10 images are 32x32x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Prunable fully connected layers
        # After 3 pooling operations: 32 -> 16 -> 8 -> 4
        # Feature size: 128 * 4 * 4 = 2048
        self.fc1 = PrunableLinear(128 * 4 * 4, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)  # 10 classes in CIFAR-10
    
    def forward(self, x):
        """Forward pass through the network"""
        # Convolutional feature extraction
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Prunable FC layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def get_all_gates(self):
        """Collect all gate values from all prunable layers"""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates().detach().cpu().flatten())
        return torch.cat(gates)
    
    def compute_sparsity_loss(self):
        """
        Compute L1 penalty on all gates to encourage sparsity.
        
        L1 norm encourages values to go to exactly 0, creating sparse networks.
        """
        sparsity_loss = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates()
                # L1 norm: sum of absolute values (gates are already positive)
                sparsity_loss += gates.sum()
        return sparsity_loss


def get_data_loaders(batch_size=128):
    """
    Prepare CIFAR-10 data loaders with standard augmentation.
    
    Returns:
        train_loader, test_loader
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download and load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, lambda_sparsity):
    """
    Train for one epoch.
    
    Args:
        model: The neural network
        train_loader: Training data loader
        optimizer: Optimizer for parameter updates
        criterion: Classification loss function
        device: cuda or cpu
        lambda_sparsity: Coefficient for sparsity regularization
        
    Returns:
        Average loss, Average accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Classification loss
        classification_loss = criterion(outputs, labels)
        
        # Sparsity loss (L1 penalty on gates)
        sparsity_loss = model.compute_sparsity_loss()
        
        # Total loss = Classification loss + λ * Sparsity loss
        total_loss = classification_loss + lambda_sparsity * sparsity_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test set.
    
    Returns:
        Average loss, Accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(test_loader), 100. * correct / total


def calculate_sparsity(model, threshold=1e-2):
    """
    Calculate the percentage of weights that are pruned.
    
    A weight is considered pruned if its gate value is below the threshold.
    
    Returns:
        Sparsity percentage (0-100)
    """
    total_params = 0
    pruned_params = 0
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates()
            total_params += gates.numel()
            pruned_params += (gates < threshold).sum().item()
    
    return 100.0 * pruned_params / total_params if total_params > 0 else 0.0


def plot_gate_distribution(model, lambda_val, save_path='gate_distribution.png'):
    """
    Plot the distribution of gate values to visualize pruning.
    
    A successful model shows a spike near 0 (pruned) and a cluster away from 0 (active).
    """
    gates = model.get_all_gates().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Gate Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Gate Values (λ = {lambda_val})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gate distribution plot saved to {save_path}")


def train_and_evaluate(lambda_sparsity, num_epochs=50, device='cuda'):
    """
    Complete training and evaluation pipeline for a given lambda value.
    
    Args:
        lambda_sparsity: Sparsity regularization coefficient
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Training with λ = {lambda_sparsity}")
    print(f"{'='*60}\n")
    
    # Prepare data
    train_loader, test_loader = get_data_loaders(batch_size=128)
    
    # Initialize model
    model = SelfPruningNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, lambda_sparsity
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    # Calculate final sparsity
    sparsity = calculate_sparsity(model, threshold=1e-2)
    
    print(f"\n{'='*60}")
    print(f"Final Results for λ = {lambda_sparsity}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Sparsity Level: {sparsity:.2f}%")
    print(f"{'='*60}\n")
    
    # Save model and plot
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), f'results/model_lambda_{lambda_sparsity}.pth')
    plot_gate_distribution(model, lambda_sparsity, f'results/gates_lambda_{lambda_sparsity}.png')
    
    return {
        'lambda': lambda_sparsity,
        'test_accuracy': best_acc,
        'sparsity': sparsity,
        'model': model
    }


def main():
    """Main function to run experiments with different lambda values"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test with different lambda values
    # Low, Medium, High sparsity regularization
    lambda_values = [0.0001, 0.001, 0.01]
    
    results = []
    for lambda_val in lambda_values:
        result = train_and_evaluate(lambda_val, num_epochs=50, device=device)
        results.append(result)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Lambda':<15} {'Test Accuracy (%)':<20} {'Sparsity Level (%)':<20}")
    print("-"*80)
    for r in results:
        print(f"{r['lambda']:<15} {r['test_accuracy']:<20.2f} {r['sparsity']:<20.2f}")
    print("="*80)
    
    # Create a comparison plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    lambdas = [r['lambda'] for r in results]
    accuracies = [r['test_accuracy'] for r in results]
    plt.plot(lambdas, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Lambda (λ)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Sparsity Regularization', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sparsities = [r['sparsity'] for r in results]
    plt.plot(lambdas, sparsities, 's-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Lambda (λ)', fontsize=12)
    plt.ylabel('Sparsity Level (%)', fontsize=12)
    plt.title('Sparsity vs Regularization Strength', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to results/comparison_plot.png")


if __name__ == "__main__":
    main()
