import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from geometrize.losses import vae_loss, hausdorff_distance
from geometrize.token_map import TokenMap
from geometrize.config import Config

def test_vae_loss():
    """Test the VAE loss function"""
    print("\n=== Testing VAE Loss ===")
    
    # Create fake data
    batch_size = 4
    seq_length = 10
    token_dim = len(TokenMap().token_to_id)
    num_bins = 20
    
    # Generate random token probabilities (logits)
    token_probs = torch.randn(batch_size, seq_length, token_dim)
    bin_probs = torch.randn(batch_size, seq_length, num_bins)
    
    # Generate target tokens and bins
    target_tokens = torch.randint(0, token_dim, (batch_size, seq_length))
    target_bins = torch.randint(0, num_bins, (batch_size, seq_length))
    
    # Add some NULL tokens for padding
    null_token_id = TokenMap().get_id('NULL')
    target_tokens[:, -2:] = null_token_id
    
    # Create latent variables
    latent_dim = 256
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    # Calculate loss
    loss = vae_loss(token_probs, bin_probs, target_tokens, target_bins, mu, logvar)
    
    print(f"VAE Loss: {loss.item()}")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss).any(), "Loss contains NaN values"
    
    # Test with different beta values
    beta_values = [0.01, 0.1, 1.0, 5.0]
    losses = []
    
    for beta in beta_values:
        # Monkey patch the beta value in the loss function (not ideal, but works for testing)
        original_beta = Config.beta
        Config.beta = beta
        
        loss = vae_loss(token_probs, bin_probs, target_tokens, target_bins, mu, logvar)
        losses.append(loss.item())
        
        # Reset beta
        Config.beta = original_beta
    
    # Plot loss vs beta
    plt.figure(figsize=(8, 5))
    plt.plot(beta_values, losses, marker='o')
    plt.xscale('log')
    plt.xlabel('Beta Value')
    plt.ylabel('Loss')
    plt.title('VAE Loss vs Beta Value')
    plt.grid(True)
    
    # Save the plot
    os.makedirs(Config.output_dir, exist_ok=True)
    plt.savefig(os.path.join(Config.output_dir, 'vae_loss_vs_beta.png'))
    
    print("VAE loss test passed!")

def test_hausdorff_distance():
    """Test the Hausdorff distance implementation"""
    print("\n=== Testing Hausdorff Distance ===")
    
    # Create two simple point clouds
    batch_size = 2
    n_points = 100
    
    # First point cloud: points on a sphere
    theta = np.random.uniform(0, np.pi, (batch_size, n_points))
    phi = np.random.uniform(0, 2*np.pi, (batch_size, n_points))
    
    x1 = np.sin(theta) * np.cos(phi)
    y1 = np.sin(theta) * np.sin(phi)
    z1 = np.cos(theta)
    
    points1 = torch.tensor(np.stack([x1, y1, z1], axis=2), dtype=torch.float)
    
    # Second point cloud: slightly perturbed sphere
    perturbation = 0.1
    points2 = points1 + perturbation * torch.randn_like(points1)
    
    # Calculate Hausdorff distance
    distance = hausdorff_distance(points1, points2)
    
    print(f"Hausdorff distance: {distance}")
    assert distance.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {distance.shape}"
    
    # Test with different perturbation levels
    perturbation_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    distances = []
    
    for perturb in perturbation_levels:
        points2 = points1 + perturb * torch.randn_like(points1)
        distance = hausdorff_distance(points1, points2)
        distances.append(distance.mean().item())
    
    # Plot distance vs perturbation
    plt.figure(figsize=(8, 5))
    plt.plot(perturbation_levels, distances, marker='o')
    plt.xlabel('Perturbation Level')
    plt.ylabel('Average Hausdorff Distance')
    plt.title('Hausdorff Distance vs Perturbation Level')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(Config.output_dir, 'hausdorff_distance_vs_perturbation.png'))
    
    print("Hausdorff distance test passed!")

def run_all_tests():
    """Run all loss function tests"""
    # Make sure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Test VAE loss
    test_vae_loss()
    
    # Test Hausdorff distance
    test_hausdorff_distance()
    
    print("\nAll loss function tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
