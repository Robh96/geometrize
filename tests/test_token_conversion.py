import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from token_map import TokenMap
from shape_generator import (
    OperationType, PrimitiveType, TransformType, BooleanType,
    generate_shape_from_tokens
)
from pipeline import convert_tokens_to_operations
from config import Config
import os
import random

def visualize_point_cloud(points, title="Generated Shape"):
    """Visualize a point cloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors based on coordinate values for better visualization
    colors = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save as image
    os.makedirs(Config.output_dir, exist_ok=True)
    file_path = os.path.join(Config.output_dir, f"{title.replace(' ', '_')}.png")
    
    # Plot using matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample points for clarity if needed
    if len(points) > 500:
        indices = np.random.choice(len(points), 500, replace=False)
        plot_points = points[indices]
    else:
        plot_points = points
    
    # Plot the points
    ax.scatter(
        plot_points[:, 0],
        plot_points[:, 1], 
        plot_points[:, 2],
        c=plot_points[:, 2],  # Color by z-coordinate
        cmap='viridis',
        s=20,
        alpha=0.7
    )
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Save figure
    plt.savefig(file_path)
    plt.close()
    
    print(f"Visualization saved as {file_path}")
    return pcd

def test_sphere_token_sequence():
    """Test creating a sphere from tokens"""
    print("\n=== Testing Sphere Token Sequence ===")
    
    token_map = TokenMap()
    
    # Create a sequence for a sphere
    tokens = np.array([
        token_map.get_id('SPHERE'),
        token_map.get_id('CENTER_X'), 
        10,  # bin value
        token_map.get_id('CENTER_Y'), 
        10,  # bin value
        token_map.get_id('CENTER_Z'), 
        10,  # bin value
        token_map.get_id('RADIUS'), 
        10,  # bin value
        token_map.get_id('NULL')
    ])
    
    bins = np.array([0, 0, 10, 0, 10, 0, 10, 0, 10, 0])
    
    # Convert tokens to operations
    operations = convert_tokens_to_operations(tokens, bins)
    
    # Print the operations
    print("Generated operations:")
    for i, op in enumerate(operations):
        print(f"Operation {i}:")
        for k, v in op.items():
            if k == 'parameters':
                print(f"  {k}:")
                for param_k, param_v in v.items():
                    print(f"    {param_k}: {param_v}")
            else:
                print(f"  {k}: {v}")
    
    # Generate shape from operations
    points = generate_shape_from_tokens(operations)
    
    # Visualize the point cloud
    visualize_point_cloud(points, "Sphere from Tokens")
    
    print("Sphere token sequence test passed!")
    return operations

def test_complex_token_sequence():
    """Test a more complex token sequence with transformations and boolean operations"""
    print("\n=== Testing Complex Token Sequence ===")
    
    token_map = TokenMap()
    
    # Create a sequence for sphere and cube with boolean union
    tokens = np.array([
        # First create a sphere
        token_map.get_id('SPHERE'),
        token_map.get_id('CENTER_X'), 
        10,  # bin value
        token_map.get_id('CENTER_Y'), 
        10,  # bin value
        token_map.get_id('CENTER_Z'), 
        5,   # bin value
        token_map.get_id('RADIUS'), 
        8,   # bin value
        
        # Then create a cube
        token_map.get_id('CUBOID'),
        token_map.get_id('CENTER_X'), 
        10,  # bin value
        token_map.get_id('CENTER_Y'), 
        10,  # bin value
        token_map.get_id('CENTER_Z'), 
        15,  # bin value
        token_map.get_id('DX'), 
        8,   # bin value
        token_map.get_id('DY'), 
        8,   # bin value
        token_map.get_id('DZ'), 
        8,   # bin value
        
        # Union them
        token_map.get_id('UNION'),
        token_map.get_id('OBJECT_1'), 
        1,   # bin value
        token_map.get_id('OBJECT_2'), 
        2,   # bin value
        
        token_map.get_id('NULL')
    ])
    
    # Create bin values (not all will be used)
    bins = np.array([
        0, 0, 10, 0, 10, 0, 5, 0, 8,  # Sphere
        0, 0, 10, 0, 10, 0, 15, 0, 8, 0, 8, 0, 8,  # Cube
        0, 0, 1, 0, 2,  # Union
        0
    ])
    
    # Convert tokens to operations
    operations = convert_tokens_to_operations(tokens, bins)
    
    # Print the operations
    print("Generated operations:")
    for i, op in enumerate(operations):
        print(f"Operation {i}:")
        for k, v in op.items():
            if k == 'parameters':
                print(f"  {k}:")
                for param_k, param_v in v.items():
                    print(f"    {param_k}: {param_v}")
            else:
                print(f"  {k}: {v}")
    
    # Generate shape from operations
    points = generate_shape_from_tokens(operations)
    
    # Visualize the point cloud
    visualize_point_cloud(points, "Complex Shape from Tokens")
    
    print("Complex token sequence test passed!")
    return operations

def test_random_token_sequences():
    """Test conversion of random token sequences"""
    print("\n=== Testing Random Token Sequences ===")
    
    token_map = TokenMap()
    num_sequences = 3
    max_seq_length = 30
    
    for i in range(num_sequences):
        # Generate a random sequence length
        seq_length = random.randint(5, max_seq_length)
        
        # Generate random token IDs
        tokens = torch.randint(0, len(token_map.token_to_id), (seq_length,))
        bins = torch.randint(0, Config.num_bins, (seq_length,))
        
        print(f"\nRandom sequence {i+1}:")
        print("Tokens:", [token_map.get_token(t.item()) for t in tokens])
        print("Bins:", bins.numpy())
        
        # Try to convert to operations
        try:
            operations = convert_tokens_to_operations(tokens.numpy(), bins.numpy())
            print(f"Converted to {len(operations)} operations")
            
            # Try to generate shape
            if operations:
                try:
                    points = generate_shape_from_tokens(operations)
                    if points is not None:
                        visualize_point_cloud(points, f"Random_Sequence_{i+1}")
                        print("Successfully generated shape")
                    else:
                        print("No points generated")
                except Exception as e:
                    print(f"Error generating shape: {e}")
            else:
                print("No operations generated")
            
        except Exception as e:
            print(f"Error converting tokens to operations: {e}")
    
    print("Random token sequence tests completed!")

def run_all_tests():
    """Run all token conversion tests"""
    # Make sure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Test simple sphere token sequence
    test_sphere_token_sequence()
    
    # Test a more complex token sequence
    test_complex_token_sequence()
    
    # Test random token sequences
    test_random_token_sequences()
    
    print("\nAll token conversion tests completed!")

if __name__ == "__main__":
    run_all_tests()
