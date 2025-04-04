import numpy as np
from stl import mesh
import open3d as o3d

def load_stl_to_pointcloud(file_path, n_points=2048):
    """Load an STL file and convert to a normalized point cloud."""
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)
    
    # Extract points from the mesh
    points = np.vstack([stl_mesh.v0, stl_mesh.v1, stl_mesh.v2])
    
    # Normalize the point cloud
    points = normalize_points(points)
    
    # Downsample to a fixed number of points
    points = downsample_points(points, n_points)
    
    return points

def normalize_points(points):
    """Center the point cloud around origin and scale to unit cube."""
    # Center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale
    max_dist = np.max(np.abs(points))
    points = points / max_dist
    
    return points

def downsample_points(points, n_points):
    """Downsample the point cloud to n_points using adaptive methods."""
    # Use Open3D for downsampling
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # If we already have fewer points than requested
    if len(points) <= n_points:
        return points
    
    # Try farthest point sampling first (gives better distribution)
    try:
        downsampled_pcd = pcd.farthest_point_down_sample(n_points)
        downsampled_points = np.asarray(downsampled_pcd.points)
        return downsampled_points
    except Exception as e:
        # Fall back to voxel downsampling if FPS fails
        try:
            # Calculate voxel size based on bounding box and desired point count
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_volume = np.prod(bbox.get_extent())
            voxel_size = (bbox_volume / n_points) ** (1/3)
            
            downsampled_pcd = pcd.voxel_down_sample(voxel_size)
            
            # Adjust voxel size if we didn't get close enough to target point count
            downsampled_points = np.asarray(downsampled_pcd.points)
            if abs(len(downsampled_points) - n_points) > 0.1 * n_points:  # More than 10% off
                # Use uniform downsampling as last resort
                indices = np.random.choice(len(points), n_points, replace=False)
                downsampled_points = points[indices]
            
            return downsampled_points
        except:
            # Last resort: random sampling
            indices = np.random.choice(len(points), n_points, replace=False)
            return points[indices]