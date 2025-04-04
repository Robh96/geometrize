import os
import numpy as np
import open3d as o3d
from geometrize.data_processing import load_stl_to_pointcloud
from geometrize.config import Config
import matplotlib.pyplot as plt
import pytest
from geometrize.generate_test_shapes import save_test_shapes

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

@pytest.fixture
def stl_path():
    """Fixture that provides a path to a test STL file"""
    # Ensure we have test shapes
    os.makedirs(Config.data_dir, exist_ok=True)
    test_stl_path = os.path.join(Config.data_dir, "sphere.stl")
    
    # If the test file doesn't exist, generate it
    if not os.path.exists(test_stl_path):
        save_test_shapes()
    
    return test_stl_path

def test_stl_processing(stl_path, n_points=2048):
    """Test loading and processing an STL file"""
    # Load the point cloud
    point_cloud = load_stl_to_pointcloud(stl_path, n_points)
    
    # Check basic properties
    assert point_cloud is not None, "Point cloud should not be None"
    assert isinstance(point_cloud, np.ndarray), "Point cloud should be a numpy array"
    assert point_cloud.shape == (n_points, 3), f"Point cloud should have shape ({n_points}, 3)"
    
    # Check normalization
    assert np.min(point_cloud) >= -1.0, "Point cloud should be normalized to [-1, 1]"
    assert np.max(point_cloud) <= 1.0, "Point cloud should be normalized to [-1, 1]"
    
    # Check for NaNs or Infs
    assert not np.isnan(point_cloud).any(), "Point cloud should not contain NaN values"
    assert not np.isinf(point_cloud).any(), "Point cloud should not contain Inf values"
    
    print(f"Successfully loaded point cloud from {stl_path} with {n_points} points")
    
    # Store for other tests without returning
    test_stl_processing.point_cloud = point_cloud

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