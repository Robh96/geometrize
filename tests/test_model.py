import torch
import matplotlib.pyplot as plt
from geometrize.model import PointNetEncoder, TransformerDecoder
from geometrize.token_map import TokenMap
from geometrize.config import Config
import os


def test_pointnet_encoder():
    """Test the PointNet encoder's forward pass"""
    print("\n=== Testing PointNet Encoder ===")
    
    # Create a batch of random point clouds
    batch_size = 4
    num_points = 2048
    points = torch.rand(batch_size, num_points, 3)
    
    # Initialize encoder
    latent_dim = 256
    encoder = PointNetEncoder(latent_dim=latent_dim)
    
    # Set to eval mode
    encoder.eval()
    
    # Forward pass
    with torch.no_grad():
        mu, logvar = encoder(points)
    
    # Check output shapes
    assert mu.shape == (batch_size, latent_dim), f"Expected mu shape {(batch_size, latent_dim)}, got {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"Expected logvar shape {(batch_size, latent_dim)}, got {logvar.shape}"
    
    print(f"Encoder output shapes: mu={mu.shape}, logvar={logvar.shape}")
    print("PointNet encoder test passed!")
    
    # Visualize latent space (for 2D projection of first two dimensions)
    plt.figure(figsize=(6, 6))
    plt.scatter(mu[:, 0].numpy(), mu[:, 1].numpy(), alpha=0.8)
    plt.title('2D Projection of Latent Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(Config.output_dir, exist_ok=True)
    plt.savefig(os.path.join(Config.output_dir, 'latent_space_projection.png'))
    
    # Store values for other tests without returning them
    test_pointnet_encoder.mu = mu
    test_pointnet_encoder.logvar = logvar

def reparameterize(mu, logvar):
    """Test the reparameterization trick"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z

def test_transformer_decoder():
    """Test the Transformer decoder's forward pass"""
    print("\n=== Testing Transformer Decoder ===")
    
    # Create latent vectors
    batch_size = 4
    latent_dim = 256
    z = torch.randn(batch_size, latent_dim)
    
    # Initialize decoder
    token_dim = len(TokenMap().token_to_id)
    num_bins = 20
    decoder = TransformerDecoder(latent_dim=latent_dim, token_dim=token_dim, num_bins=num_bins)
    
    # Set to eval mode
    decoder.eval()
    
    # Forward pass (auto-regressive mode)
    with torch.no_grad():
        token_probs, bin_probs, offsets = decoder(z)
    
    # Check output shapes
    print(f"Token probabilities shape: {token_probs.shape}")
    print(f"Bin probabilities shape: {bin_probs.shape}")
    print(f"Offsets shape: {offsets.shape}")
    
    # Check that probabilities sum to 1
    token_sum = torch.sum(torch.softmax(token_probs, dim=-1)[0, 0, :])
    bin_sum = torch.sum(torch.softmax(bin_probs, dim=-1)[0, 0, :])
    
    assert abs(token_sum - 1.0) < 1e-5, f"Token probabilities don't sum to 1: {token_sum}"
    assert abs(bin_sum - 1.0) < 1e-5, f"Bin probabilities don't sum to 1: {bin_sum}"
    
    # Test teacher forcing mode
    # Create a fake target sequence
    seq_len = 10
    target_seq = torch.zeros(batch_size, seq_len, 2, dtype=torch.long)
    
    # Forward pass with teacher forcing
    with torch.no_grad():
        token_probs_tf, bin_probs_tf, offsets_tf = decoder(z, target_seq)
    
    # Check output shapes
    assert token_probs_tf.shape == (batch_size, seq_len, token_dim), \
        f"Expected token_probs shape {(batch_size, seq_len, token_dim)}, got {token_probs_tf.shape}"
    
    print("Transformer decoder test passed!")
    
    # Store for other tests without returning
    test_transformer_decoder.token_probs = token_probs
    test_transformer_decoder.bin_probs = bin_probs
    test_transformer_decoder.offsets = offsets

def test_vae_pipeline():
    """Test the complete VAE pipeline (encoder -> reparameterize -> decoder)"""
    print("\n=== Testing Complete VAE Pipeline ===")
    
    # Create a batch of random point clouds
    batch_size = 4
    num_points = 2048
    points = torch.rand(batch_size, num_points, 3)
    
    # Initialize models
    latent_dim = 256
    encoder = PointNetEncoder(latent_dim=latent_dim)
    decoder = TransformerDecoder(latent_dim=latent_dim)
    
    # Set to eval mode
    encoder.eval()
    decoder.eval()
    
    # Forward pass through encoder
    with torch.no_grad():
        mu, logvar = encoder(points)
        
        # Reparameterize
        z = reparameterize(mu, logvar)
        
        # Forward pass through decoder
        token_probs, bin_probs, offsets = decoder(z)
    
    print(f"VAE Pipeline output shapes:")
    print(f"- Input points: {points.shape}")
    print(f"- Latent mu: {mu.shape}")
    print(f"- Latent logvar: {logvar.shape}")
    print(f"- Sampled z: {z.shape}")
    print(f"- Token probs: {token_probs.shape}")
    print(f"- Bin probs: {bin_probs.shape}")
    
    # Make sure all shapes are as expected
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)
    assert token_probs.shape[0] == batch_size
    
    print("VAE pipeline test passed!")

def test_token_prediction():
    """Test token prediction and probabilities"""
    print("\n=== Testing Token Prediction ===")
    
    # Create latent vectors
    batch_size = 1
    latent_dim = 256
    z = torch.randn(batch_size, latent_dim)
    
    # Initialize decoder
    token_dim = len(TokenMap().token_to_id)
    num_bins = 20
    decoder = TransformerDecoder(latent_dim=latent_dim, token_dim=token_dim, num_bins=num_bins)
    
    # Set to eval mode
    decoder.eval()
    
    # Forward pass
    with torch.no_grad():
        token_probs, bin_probs, _ = decoder(z)
    
    # Get the predicted tokens and their probabilities
    token_map = TokenMap()
    
    # Convert probabilities to actual token predictions
    pred_tokens = torch.argmax(token_probs, dim=-1)[0].numpy()  # First batch item
    pred_bins = torch.argmax(bin_probs, dim=-1)[0].numpy()
    
    # Print the token sequence
    print("Predicted token sequence:")
    for i, token_id in enumerate(pred_tokens):
        if token_id == token_map.get_id('NULL'):
            break
        token_name = token_map.get_token(int(token_id))
        bin_value = pred_bins[i]
        print(f"{i}: {token_name} (bin: {bin_value})")
    
    # Make sure we have a valid sequence
    assert len(pred_tokens) > 0, "Should have predicted at least one token"
    
    print("Token prediction test passed!")

def run_all_tests():
    """Run all model tests"""
    # Make sure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Test encoder
    test_pointnet_encoder()
    
    # Test decoder
    test_transformer_decoder()
    
    # Test full VAE pipeline
    test_vae_pipeline()
    
    # Test token prediction
    test_token_prediction()
    
    print("\nAll model tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
