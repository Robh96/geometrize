import os
import numpy as np
import open3d as o3d
from geometrize.data_processing import load_stl_to_pointcloud
from geometrize.config import Config
import matplotlib.pyplot as plt

def visualize_point_cloud(points, title="Point Cloud"):
    """Visualize a point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors based on coordinate values for better visualization
    colors = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create a coordinate frame to show origin
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0])
    
    # Visualize
    print(f"Displaying {title}")
    print(f"Number of points: {len(points)}")
    print(f"Centroid: {np.mean(points, axis=0)}")
    print(f"Min: {np.min(points, axis=0)}")
    print(f"Max: {np.max(points, axis=0)}")
    print(f"Bounding box dimensions: {np.max(points, axis=0) - np.min(points, axis=0)}")
    
    o3d.visualization.draw_geometries([pcd, frame], window_name=title)
    
    return pcd

def verify_normalization(points, tolerance=1e-6):
    """Verify that the point cloud is properly normalized."""
    # Check if centered at origin
    centroid = np.mean(points, axis=0)
    is_centered = np.allclose(centroid, np.zeros(3), atol=tolerance)
    
    # Check if scaled to unit cube
    max_distance = np.max(np.abs(points))
    is_scaled = np.isclose(max_distance, 1.0, atol=tolerance)
    
    return is_centered, is_scaled

def plot_point_cloud_stats(points):
    """Plot histograms of point distributions along x, y, z axes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, axis in enumerate(['x', 'y', 'z']):
        axes[i].hist(points[:, i], bins=50)
        axes[i].set_title(f"{axis}-axis distribution")
        axes[i].axvline(x=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, "point_distribution.png"))
    plt.show()

def test_stl_processing(stl_path, n_points=2048):
    """Test the full STL processing pipeline."""
    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Process the STL file
    print(f"Processing STL file: {stl_path}")
    try:
        points = load_stl_to_pointcloud(stl_path, n_points)
    except Exception as e:
        print(f"Error loading STL file: {e}")
        return
    
    # Verify the normalization
    is_centered, is_scaled = verify_normalization(points)
    print(f"Point cloud properly centered: {is_centered}")
    print(f"Point cloud properly scaled: {is_scaled}")
    print(f"Point cloud shape: {points.shape}")
    
    # Visualize the point cloud
    visualize_point_cloud(points, "Normalized Point Cloud")
    
    # Plot statistics
    plot_point_cloud_stats(points)
    
    return points

if __name__ == "__main__":
    # Test with a sample STL file
    # You'll need to provide a path to an STL file
    sample_stl_path = os.path.join(Config.data_dir, "sphere.stl")
    
    # Check if the file exists
    if not os.path.exists(sample_stl_path):
        print(f"Sample STL file not found at {sample_stl_path}")
        print("Please provide a valid path to an STL file:")
        sample_stl_path = input().strip()
    
    test_stl_processing(sample_stl_path, Config.num_points)