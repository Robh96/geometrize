"""
Basic usage example for Geometrize

This script demonstrates:
1. Generating test shapes
2. Loading a point cloud
3. Visualizing a point cloud
4. Running basic model inference
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the geometrize package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometrize.config import Config
from geometrize.generate_test_shapes import save_test_shapes
from geometrize.data_processing import load_stl_to_pointcloud
from geometrize.model import PointNetEncoder, TransformerDecoder
from geometrize.pipeline import reparameterize, convert_tokens_to_operations
from geometrize.shape_generator import generate_shape_from_tokens

# 1. Generate test shapes
print("Generating test shapes...")
save_test_shapes()

# 2. Load a point cloud from an STL file
print("\nLoading point cloud from STL...")
stl_path = os.path.join(Config.data_dir, "sphere.stl")
point_cloud = load_stl_to_pointcloud(stl_path, Config.num_points)
print(f"Loaded point cloud with {len(point_cloud)} points")

# 3. Visualize the point cloud
print("\nVisualizing point cloud...")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    point_cloud[:, 0],
    point_cloud[:, 1],
    point_cloud[:, 2],
    c=point_cloud[:, 2],
    cmap='viridis',
    s=5
)
ax.set_title("Sample Point Cloud")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1,1,1])

# Create output directory if it doesn't exist
os.makedirs("examples_output", exist_ok=True)
plt.savefig("examples_output/point_cloud_visualization.png")
plt.close()
print("Point cloud visualization saved to examples_output/point_cloud_visualization.png")

# 4. Set up basic model (not trained)
print("\nSetting up model (demo only)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = PointNetEncoder(latent_dim=Config.latent_dim).to(device)
decoder = TransformerDecoder(
    latent_dim=Config.latent_dim,
    token_dim=30,  # From TokenMap
    num_bins=Config.num_bins
).to(device)

# For a real application, you would load pre-trained weights:
# checkpoint = torch.load('path_to_checkpoint.pth')
# encoder.load_state_dict(checkpoint['encoder_state_dict'])
# decoder.load_state_dict(checkpoint['decoder_state_dict'])

# 5. Demonstrate inference flow (with untrained model, just for API demonstration)
print("\nDemonstrating inference flow (untrained model)...")
point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float).unsqueeze(0).to(device)

with torch.no_grad():
    # Encode point cloud to latent representation
    mu, logvar = encoder(point_cloud_tensor)
    
    # Sample from latent space
    z = reparameterize(mu, logvar)
    
    # Decode to get token and bin predictions
    token_probs, bin_probs, _ = decoder(z)
    
    # Convert to actual tokens and bins
    pred_tokens = torch.argmax(token_probs, dim=-1)[0].cpu().numpy()
    pred_bins = torch.argmax(bin_probs, dim=-1)[0].cpu().numpy()

print(f"Generated token sequence of length {len(pred_tokens)}")

# Note: In a real application with a trained model, you would:
# 1. Convert tokens to operations: operations = convert_tokens_to_operations(pred_tokens, pred_bins)
# 2. Generate shape from operations: result = generate_shape_from_tokens(operations)
# 3. Visualize or save the result

print("\nExample complete!")
