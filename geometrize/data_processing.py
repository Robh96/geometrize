import numpy as np
import stl
import os
import random
from .config import Config

def load_stl_to_pointcloud(stl_file, num_points=2048):
    """
    Load an STL file and convert it to a point cloud with a specific number of points.
    
    Args:
        stl_file: Path to the STL file
        num_points: Number of points to sample
        
    Returns:
        np.ndarray: Point cloud with shape (num_points, 3)
    """
    try:
        # Load the mesh using numpy-stl
        mesh = stl.mesh.Mesh.from_file(stl_file)
        
        # Get triangle vertices
        vertices = np.vstack([mesh.v0, mesh.v1, mesh.v2])
        
        # Sample points by weighted random selection from triangles
        points = sample_points_from_mesh(mesh, num_points)
        
        # Ensure we have exactly num_points
        if len(points) < num_points:
            # If we have too few points, duplicate some randomly
            needed = num_points - len(points)
            indices = np.random.randint(0, len(points), needed)
            extra_points = points[indices]
            # Add small jitter to avoid exact duplicates
            extra_points += np.random.normal(0, 0.001, extra_points.shape)
            points = np.vstack([points, extra_points])
        elif len(points) > num_points:
            # If we have too many points, subsample
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
            
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit cube
        max_distance = np.max(np.abs(points))
        if max_distance > 0:  # Avoid division by zero
            points = points / max_distance
            
        return points
        
    except Exception as e:
        print(f"Error loading STL file {stl_file}: {e}")
        # Return a default point cloud (sphere)
        return _generate_default_point_cloud(num_points)

def sample_points_from_mesh(mesh, num_points):
    """Sample points from an STL mesh with proper area weighting"""
    # Get triangles
    triangles = np.vstack([mesh.v0, mesh.v1, mesh.v2]).reshape(-1, 3, 3)
    
    # Calculate areas of triangles
    sides_a = triangles[:, 1] - triangles[:, 0]
    sides_b = triangles[:, 2] - triangles[:, 0]
    
    # Use cross product to get area vectors
    areas = 0.5 * np.linalg.norm(np.cross(sides_a, sides_b), axis=1)
    
    # Sample triangles with probability proportional to area
    probabilities = areas / np.sum(areas)
    triangle_indices = np.random.choice(len(triangles), size=num_points, p=probabilities)
    selected_triangles = triangles[triangle_indices]
    
    # Generate random points within each triangle
    r1 = np.random.random(num_points)
    r2 = np.random.random(num_points)
    
    # Use barycentric coordinates to sample within triangles
    # If sqrt(r1) + r2 > 1, reflect point to ensure uniform distribution
    mask = (np.sqrt(r1) + r2 > 1)
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    
    a = 1 - np.sqrt(r1)
    b = np.sqrt(r1) * (1 - r2)
    c = np.sqrt(r1) * r2
    
    # Apply barycentric coordinates
    points = (
        a[:, np.newaxis] * selected_triangles[:, 0] +
        b[:, np.newaxis] * selected_triangles[:, 1] +
        c[:, np.newaxis] * selected_triangles[:, 2]
    )
    
    return points

def _generate_default_point_cloud(num_points):
    """Generate a default point cloud (sphere) when loading fails"""
    # Using Fibonacci sphere algorithm for uniform distribution
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    points = np.column_stack((x, y, z))
    return points