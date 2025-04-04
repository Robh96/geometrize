import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from torch.utils.data import DataLoader

from geometrize.pipeline import (
    ShapeDataset, reparameterize, load_or_generate_data,
    generate_synthetic_tokens, convert_tokens_to_operations
)
from geometrize.model import PointNetEncoder, TransformerDecoder
from geometrize.losses import vae_loss, hausdorff_distance, combined_loss
from geometrize.config import Config
from geometrize.token_map import TokenMap
from geometrize.shape_generator import OperationType, PrimitiveType, TransformType, BooleanType

def test_data_loading():
    """Test data loading functionality"""
    print("\n=== Testing Data Loading ===")
    
    start_time = time.time()
    data_paths, token_sequences = load_or_generate_data()
    end_time = time.time()
    
    print(f"Loaded {len(data_paths)} data samples in {end_time - start_time:.2f} seconds")
    print(f"First data path: {data_paths[0]}")
    print(f"First token sequence shape: {token_sequences[0].shape}")
    
    # Create dataset and dataloader
    dataset = ShapeDataset(data_paths, token_sequences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Get a batch
    point_clouds, tokens = next(iter(dataloader))
    
    print(f"Batch shapes: points={point_clouds.shape}, tokens={tokens.shape}")
    
    # Verify shapes
    assert len(data_paths) > 0, "Should have loaded at least one data sample"
    assert len(token_sequences) == len(data_paths), "Should have one token sequence per data path"
    assert point_clouds.shape[0] > 0, "Should have at least one point cloud in batch"
    assert tokens.shape[0] == point_clouds.shape[0], "Should have same batch size for points and tokens"
    
    # Visualize first point cloud
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    points = point_clouds[0].numpy()
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=points[:, 2],
        cmap='viridis',
        s=5,
        alpha=0.6
    )
    
    ax.set_title("Sample Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])
    
    # Save the figure
    os.makedirs(Config.output_dir, exist_ok=True)
    plt.savefig(os.path.join(Config.output_dir, 'sample_point_cloud.png'))
    plt.close()
    
    print("Data loading test passed!")
    
    # Store results for other tests without returning
    test_data_loading.data_paths = data_paths
    test_data_loading.token_sequences = token_sequences

def test_token_generation():
    """Test synthetic token generation"""
    print("\n=== Testing Token Generation ===")
    
    # Generate some token sequences
    num_sequences = 5
    token_sequences = [generate_synthetic_tokens() for _ in range(num_sequences)]
    
    token_map = TokenMap()
    
    # Print information about the sequences
    for i, seq in enumerate(token_sequences):
        tokens, bins = seq[:, 0], seq[:, 1]
        print(f"Sequence {i+1}: length={len(tokens)}")
        print("  First few tokens:", [token_map.get_token(int(t)) for t in tokens[:5]])
        print("  First few bins:", bins[:5].numpy())
    
    # Verify sequences
    assert len(token_sequences) == num_sequences, "Should have generated the requested number of sequences"
    for seq in token_sequences:
        assert seq.shape[1] == 2, "Each sequence should have 2 columns (tokens and bins)"
        assert len(seq) > 0, "Sequences should not be empty"
    
    print("Token generation test passed!")
    
    # Store results for other tests without returning
    test_token_generation.token_sequences = token_sequences

def test_token_to_operation_conversion():
    """Test conversion of tokens to shape operations"""
    print("\n=== Testing Token to Operation Conversion ===")
    
    token_map = TokenMap()
    
    # Create a valid token sequence for a sphere
    tokens = np.array([
        token_map.get_id('SPHERE'),         # 0: SPHERE token
        token_map.get_id('CENTER_X'), 0,    # 1-2: CENTER_X token and bin
        token_map.get_id('CENTER_Y'), 0,    # 3-4: CENTER_Y token and bin  
        token_map.get_id('CENTER_Z'), 0,    # 5-6: CENTER_Z token and bin
        token_map.get_id('RADIUS'), 10,     # 7-8: RADIUS token and bin
        token_map.get_id('NULL')            # 9: NULL token
    ])
    
    # Corresponding bin values (now properly aligned with tokens)
    bins = np.array([0, 0, 10, 0, 10, 0, 10, 0, 10, 0])
    
    # Convert tokens to operations
    operations = convert_tokens_to_operations(tokens, bins)
    
    print(f"Generated {len(operations)} operations")
    for i, op in enumerate(operations):
        print(f"Operation {i}:")
        for key, value in op.items():
            if key == 'parameters':
                print(f"  {key}:")
                for param_key, param_value in value.items():
                    print(f"    {param_key}: {param_value}")
            else:
                print(f"  {key}: {value}")
    
    # Verify operations
    assert len(operations) > 0, "Should have generated at least one operation"
    assert 'operation' in operations[0], "Operation should have an 'operation' key"
    
    print("Token to operation conversion test passed!")
    
    # Store results for other tests without returning
    test_token_to_operation_conversion.operations = operations

def test_reparameterization():
    """Test reparameterization trick for VAE"""
    print("\n=== Testing Reparameterization ===")
    
    batch_size = 4
    latent_dim = 256
    
    # Create sample mu and logvar tensors
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    # Apply reparameterization multiple times
    num_samples = 10
    samples = []
    for _ in range(num_samples):
        z = reparameterize(mu, logvar)
        samples.append(z)
    
    # Check shapes
    assert z.shape == (batch_size, latent_dim), f"Expected z shape {(batch_size, latent_dim)}, got {z.shape}"
    
    # Check that samples are different (stochastic)
    z1 = samples[0]
    z2 = samples[1]
    diff = torch.mean(torch.abs(z1 - z2))
    
    print(f"Mean absolute difference between samples: {diff.item():.6f}")
    assert diff > 0, "Reparameterization samples should be different"
    
    # Visualize some latent points in 2D
    plt.figure(figsize=(8, 8))
    for i in range(num_samples):
        plt.scatter(samples[i][:, 0].numpy(), samples[i][:, 1].numpy(), alpha=0.5, 
                   label=f"Sample {i+1}")
    plt.plot(mu[:, 0].numpy(), mu[:, 1].numpy(), 'rx', markersize=10, label="Means")
    
    plt.title("Reparameterization Samples")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.output_dir, 'reparameterization_samples.png'))
    plt.close()
    
    print("Reparameterization test passed!")
    
    # Store results for other tests without returning
    test_reparameterization.samples = samples

def test_mini_training_loop():
    """Test a mini training loop to ensure everything works together"""
    print("\n=== Testing Mini Training Loop ===")
    
    # Get some data
    data_paths, token_sequences = load_or_generate_data()
    
    # Take a small subset for quick testing
    subset_size = min(8, len(data_paths))
    data_paths = data_paths[:subset_size]
    token_sequences = token_sequences[:subset_size]
    
    # Create dataset and dataloader
    dataset = ShapeDataset(data_paths, token_sequences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    encoder = PointNetEncoder(latent_dim=Config.latent_dim).to(device)
    decoder = TransformerDecoder(
        latent_dim=Config.latent_dim,
        token_dim=len(TokenMap().token_to_id),
        num_bins=Config.num_bins
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=Config.learning_rate
    )
    
    # Mini training loop
    num_epochs = 2
    batch_losses = []
    token_losses = []
    bin_losses = []
    kl_losses = []
    shape_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_token_loss = 0
        epoch_bin_loss = 0
        epoch_kl_loss = 0
        epoch_shape_loss = 0
        start_time = time.time()
        
        for point_clouds, token_seqs in dataloader:
            # Move data to device
            point_clouds = point_clouds.to(device)
            token_seqs = token_seqs.to(device)
            
            # Extract token and bin values
            target_tokens = token_seqs[:, :, 0]
            target_bins = token_seqs[:, :, 1]
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            mu, logvar = encoder(point_clouds)
            z = reparameterize(mu, logvar)
            token_probs, bin_probs, offsets = decoder(z, token_seqs)
            
            # Compute loss with component breakdown
            loss, (token_loss, bin_loss, kl_loss, shape_loss) = vae_loss(
                token_probs, bin_probs, target_tokens, target_bins, mu, logvar
            )
            
            # Track individual losses
            batch_losses.append(loss.item())
            token_losses.append(token_loss.item())
            bin_losses.append(bin_loss.item())
            kl_losses.append(kl_loss.item())
            shape_losses.append(shape_loss.item() if hasattr(shape_loss, 'item') else 0)
            
            # Every other batch, try to add shape loss (for testing)
            if len(batch_losses) % 2 == 0:
                try:
                    # Get predicted tokens and bins
                    pred_tokens = torch.argmax(token_probs, dim=-1)
                    pred_bins = torch.argmax(bin_probs, dim=-1)
                    
                    # Just for testing, treat original point cloud as generated for simplicity
                    # In a real case we would generate a point cloud from the tokens
                    perturbed_points = point_clouds + 0.1 * torch.randn_like(point_clouds)
                    
                    # Calculate combined loss with shape component
                    shape_weight = 0.2
                    combined, _ = combined_loss(
                        token_probs, bin_probs, target_tokens, target_bins, mu, logvar,
                        point_clouds, perturbed_points, shape_weight
                    )
                    
                    # Use the combined loss instead
                    loss = combined
                    
                except Exception as e:
                    print(f"Warning: Error calculating shape loss: {e}")
            
            # Backpropagation
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Add batch loss
            epoch_loss += loss.item()
            epoch_token_loss += token_loss.item()
            epoch_bin_loss += bin_loss.item()
            epoch_kl_loss += kl_loss.item()
            if hasattr(shape_loss, 'item'):
                epoch_shape_loss += shape_loss.item()
        
        # Report epoch stats
        epoch_time = time.time() - start_time
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_token_loss = epoch_token_loss / len(dataloader)
        avg_bin_loss = epoch_bin_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        avg_shape_loss = epoch_shape_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        print(f"  Token: {avg_token_loss:.4f}, Bin: {avg_bin_loss:.4f}, KL: {avg_kl_loss:.4f}, Shape: {avg_shape_loss:.4f}")
    
    # Verify losses
    assert len(batch_losses) > 0, "Should have recorded some batch losses"
    assert all(not np.isnan(loss) for loss in batch_losses), "Losses should not be NaN"
    assert all(not np.isnan(loss) for loss in token_losses), "Token losses should not be NaN"
    assert all(not np.isnan(loss) for loss in bin_losses), "Bin losses should not be NaN"
    assert all(not np.isnan(loss) for loss in kl_losses), "KL losses should not be NaN"
    
    # Plot loss curve with components
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(batch_losses, label='Total Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(token_losses, label='Token Loss')
    plt.plot(bin_losses, label='Bin Loss')
    plt.plot(kl_losses, label='KL Loss')
    plt.plot(shape_losses, label='Shape Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss Component')
    plt.title('Loss Components')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, 'mini_training_loss_components.png'))
    plt.close()
    
    print("Mini training loop test passed!")
    
    # Try inference with the trained model
    with torch.no_grad():
        # Get a sample
        point_cloud, _ = dataset[0]
        point_cloud = point_cloud.unsqueeze(0).to(device)  # Add batch dimension
        
        # Encode and decode
        mu, logvar = encoder(point_cloud)
        z = reparameterize(mu, logvar)
        token_probs, bin_probs, _ = decoder(z)
        
        # Get predicted tokens
        pred_tokens = torch.argmax(token_probs, dim=-1).cpu().numpy()[0]  # First batch item
        pred_bins = torch.argmax(bin_probs, dim=-1).cpu().numpy()[0]
        
        # Print tokens
        token_map = TokenMap()
        print("\nPredicted token sequence:")
        
        # Fix: Only print up to the actual length of the prediction or 10 tokens, whichever is smaller
        token_count = min(10, len(pred_tokens))
        for i in range(token_count):
            token_name = token_map.get_token(int(pred_tokens[i]))
            # Make sure we also have valid bin values
            if i < len(pred_bins):
                print(f"{i}: {token_name} (bin: {pred_bins[i]})")
            else:
                print(f"{i}: {token_name} (bin: N/A)")
    
    # Store results without returning
    test_mini_training_loop.encoder = encoder
    test_mini_training_loop.decoder = decoder
    test_mini_training_loop.batch_losses = batch_losses
    test_mini_training_loop.component_losses = (token_losses, bin_losses, kl_losses, shape_losses)

def test_loss_component_tracking():
    """Test that loss component tracking works correctly"""
    print("\n=== Testing Loss Component Tracking ===")
    
    # Create sample data
    batch_size = 2
    seq_length = 8
    token_dim = len(TokenMap().token_to_id)
    num_bins = Config.num_bins
    latent_dim = Config.latent_dim
    
    # Sample tensors
    token_probs = torch.randn(batch_size, seq_length, token_dim)
    bin_probs = torch.randn(batch_size, seq_length, num_bins)
    target_tokens = torch.randint(0, token_dim, (batch_size, seq_length))
    target_bins = torch.randint(0, num_bins, (batch_size, seq_length))
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    # Calculate VAE loss with components
    loss, (token_loss, bin_loss, kl_loss, shape_loss) = vae_loss(
        token_probs, bin_probs, target_tokens, target_bins, mu, logvar
    )
    
    print(f"Loss components - Total: {loss.item()}, Token: {token_loss.item()}, "
          f"Bin: {bin_loss.item()}, KL: {kl_loss.item()}")
    
    # Make sure each component contributes to the total
    assert token_loss > 0, "Token loss should be positive"
    assert bin_loss > 0, "Bin loss should be positive"
    assert kl_loss != 0, "KL loss should not be zero"
    
    # Verify beta weighting is applied correctly
    beta = Config.beta if hasattr(Config, 'beta') else 0.1
    expected_loss = token_loss + bin_loss + beta * kl_loss
    assert torch.isclose(loss, expected_loss), f"Expected {expected_loss.item()}, got {loss.item()}"
    
    # Create a dummy point cloud for shape loss testing
    points1 = torch.randn(batch_size, 100, 3)
    points2 = points1 + 0.1 * torch.randn_like(points1)
    
    # Test combined loss with shape component
    shape_weight = 0.5
    combined, (t_loss, b_loss, k_loss, s_loss) = combined_loss(
        token_probs, bin_probs, target_tokens, target_bins, mu, logvar,
        points1, points2, shape_weight
    )
    
    print(f"Combined loss components - Total: {combined.item()}, Shape: {s_loss.item()}")
    
    # Verify shape loss is included correctly
    assert s_loss > 0, "Shape loss should be positive"
    expected_combined = token_loss + bin_loss + beta * kl_loss + shape_weight * s_loss
    assert torch.isclose(combined, expected_combined, rtol=1e-5), \
           f"Expected combined {expected_combined.item()}, got {combined.item()}"
    
    print("Loss component tracking test passed!")

def run_all_tests():
    """Run all pipeline tests"""
    # Make sure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Test data loading
    test_data_loading()
    
    # Test token generation
    test_token_generation()
    
    # Test token to operation conversion
    test_token_to_operation_conversion()
    
    # Test reparameterization
    test_reparameterization()
    
    # Test loss component tracking (add this new test)
    test_loss_component_tracking()
    
    # Test mini training loop
    test_mini_training_loop()
    
    print("\nAll pipeline tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
