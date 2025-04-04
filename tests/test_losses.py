import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from geometrize.losses import vae_loss, hausdorff_distance, combined_loss
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
    loss, (token_loss, bin_loss, kl_loss, shape_loss) = vae_loss(
        token_probs, bin_probs, target_tokens, target_bins, mu, logvar
    )
    
    print(f"VAE Loss: {loss.item()}")
    print(f"Component losses - Token: {token_loss.item()}, Bin: {bin_loss.item()}, KL: {kl_loss.item()}")
    
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss).any(), "Loss contains NaN values"
    
    # Verify that individual components sum to total (accounting for beta weight)
    beta = Config.beta if hasattr(Config, 'beta') else 0.1
    expected_sum = token_loss + bin_loss + beta * kl_loss
    assert torch.isclose(loss, expected_sum), f"Sum of components ({expected_sum.item()}) should equal total loss ({loss.item()})"
    
    # Test with different beta values
    beta_values = [0.01, 0.1, 1.0, 5.0]
    losses = []
    token_losses = []
    bin_losses = []
    kl_losses = []
    
    for beta in beta_values:
        # Monkey patch the beta value in the loss function
        original_beta = Config.beta if hasattr(Config, 'beta') else None
        Config.beta = beta
        
        loss, components = vae_loss(token_probs, bin_probs, target_tokens, target_bins, mu, logvar)
        losses.append(loss.item())
        token_losses.append(components[0].item())
        bin_losses.append(components[1].item())
        kl_losses.append(components[2].item())
        
        # Reset beta
        if original_beta is not None:
            Config.beta = original_beta
        else:
            delattr(Config, 'beta')
    
    # Plot loss vs beta
    plt.figure(figsize=(10, 6))
    plt.plot(beta_values, losses, marker='o', label='Total Loss')
    plt.plot(beta_values, token_losses, marker='s', label='Token Loss')
    plt.plot(beta_values, bin_losses, marker='^', label='Bin Loss')
    plt.plot(beta_values, [beta * kl for beta, kl in zip(beta_values, kl_losses)], marker='x', label='Weighted KL Loss')
    
    plt.xscale('log')
    plt.xlabel('Beta Value')
    plt.ylabel('Loss')
    plt.title('VAE Loss Components vs Beta Value')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs(Config.output_dir, exist_ok=True)
    plt.savefig(os.path.join(Config.output_dir, 'vae_loss_components_vs_beta.png'))
    
    print("VAE loss test passed!")

def test_combined_loss():
    """Test the combined loss function with both VAE and shape components"""
    print("\n=== Testing Combined Loss ===")
    
    # Create fake data for VAE loss
    batch_size = 4
    seq_length = 10
    token_dim = len(TokenMap().token_to_id)
    num_bins = 20
    
    token_probs = torch.randn(batch_size, seq_length, token_dim)
    bin_probs = torch.randn(batch_size, seq_length, num_bins)
    target_tokens = torch.randint(0, token_dim, (batch_size, seq_length))
    target_bins = torch.randint(0, num_bins, (batch_size, seq_length))
    
    latent_dim = 256
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    # Create fake point clouds for shape loss
    n_points = 100
    original_points = torch.randn(batch_size, n_points, 3)
    generated_points = original_points + 0.1 * torch.randn_like(original_points)
    
    # Test combined loss with different shape weights
    shape_weights = [0, 0.1, 0.5, 1.0, 2.0]
    total_losses = []
    shape_losses = []
    
    for weight in shape_weights:
        # Set shape weight in config
        original_weight = Config.shape_weight if hasattr(Config, 'shape_weight') else None
        Config.shape_weight = weight
        
        # Calculate combined loss
        loss, components = combined_loss(
            token_probs, bin_probs, target_tokens, target_bins, mu, logvar,
            original_points, generated_points, shape_weight=weight
        )
        
        token_loss, bin_loss, kl_loss, shape_loss = components
        
        # Store results
        total_losses.append(loss.item())
        shape_losses.append(shape_loss.item())
        
        print(f"Shape weight: {weight}, Total loss: {loss.item()}, Shape loss component: {shape_loss.item()}")
        
        # Verify that loss increases with shape weight
        if weight > 0:
            assert shape_loss > 0, "Shape loss should be positive"
        
        # Reset shape weight
        if original_weight is not None:
            Config.shape_weight = original_weight
        else:
            if hasattr(Config, 'shape_weight'):
                delattr(Config, 'shape_weight')
    
    # Plot losses vs shape weight
    plt.figure(figsize=(10, 6))
    plt.plot(shape_weights, total_losses, marker='o', label='Total Loss')
    plt.plot(shape_weights, shape_losses, marker='s', label='Shape Loss Component')
    
    plt.xlabel('Shape Weight')
    plt.ylabel('Loss')
    plt.title('Combined Loss vs Shape Weight')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(Config.output_dir, 'combined_loss_vs_shape_weight.png'))
    
    # Test without point clouds
    loss_no_points, components = combined_loss(
        token_probs, bin_probs, target_tokens, target_bins, mu, logvar
    )
    
    print(f"Loss without point clouds: {loss_no_points.item()}")
    assert components[3].item() == 0, "Shape loss should be zero when no point clouds are provided"
    
    print("Combined loss test passed!")

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
    
    # Test combined loss
    test_combined_loss()
    
    # Test Hausdorff distance
    test_hausdorff_distance()
    
    print("\nAll loss function tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
