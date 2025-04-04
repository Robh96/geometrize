import numpy as np
import open3d as o3d
from build123d import *
import math
import tempfile
import os
from enum import Enum, auto
import random

class OperationType(Enum):
    """Enum for different operation types"""
    PRIMITIVE = auto()
    TRANSFORM = auto()
    BOOLEAN = auto()

class PrimitiveType(Enum):
    """Enum for primitive shape types"""
    CUBE = auto()
    SPHERE = auto()
    CYLINDER = auto()
    CONE = auto()
    TORUS = auto()

class TransformType(Enum):
    """Enum for transformation types"""
    TRANSLATE = auto()
    ROTATE = auto()
    SCALE = auto()

class BooleanType(Enum):
    """Enum for boolean operations"""
    UNION = auto()
    DIFFERENCE = auto()
    INTERSECTION = auto()

class ShapeGenerator:
    """Class for generating shapes from tokens using build123d"""
    
    def __init__(self):
        """Initialize an empty shape stack"""
        self.shape_stack = []
        
    def execute_token(self, token):
        """Execute an operation based on token values"""
        op_type = token.get('operation')
        
        if op_type == OperationType.PRIMITIVE:
            self._create_primitive(token)
        elif op_type == OperationType.TRANSFORM:
            self._apply_transform(token)
        elif op_type == OperationType.BOOLEAN:
            self._apply_boolean(token)
        
    def _create_primitive(self, token):
        """Create a primitive shape based on token parameters"""
        prim_type = token.get('primitive_type')
        params = token.get('parameters', {})
        
        if prim_type == PrimitiveType.CUBE:
            size = params.get('size', 1.0)
            shape = Box(size, size, size)
        
        elif prim_type == PrimitiveType.SPHERE:
            radius = params.get('radius', 0.5)
            shape = Sphere(radius)
        
        elif prim_type == PrimitiveType.CYLINDER:
            radius = params.get('radius', 0.5)
            height = params.get('height', 1.0)
            shape = Cylinder(radius, height)
        
        elif prim_type == PrimitiveType.CONE:
            radius = params.get('radius', 0.5)
            height = params.get('height', 1.0)
            shape = Cone(radius, height)
        
        elif prim_type == PrimitiveType.TORUS:
            major_radius = params.get('major_radius', 1.0)
            minor_radius = params.get('minor_radius', 0.25)
            shape = Torus(major_radius, minor_radius)
        
        self.shape_stack.append(shape)
    
    def _apply_transform(self, token):
        """Apply a transformation to the top shape on the stack"""
        if not self.shape_stack:
            return
            
        transform_type = token.get('transform_type')
        params = token.get('parameters', {})
        
        shape = self.shape_stack.pop()
        
        if transform_type == TransformType.TRANSLATE:
            x = params.get('x', 0.0)
            y = params.get('y', 0.0)
            z = params.get('z', 0.0)
            shape = shape.located(Location((x, y, z)))
        
        elif transform_type == TransformType.ROTATE:
            # Fix: Create proper rotation angles in degrees
            theta_x = params.get('x', 0.0)
            theta_y = params.get('y', 0.0)
            theta_z = params.get('z', 0.0)
            
            # Apply rotations in order: Z, Y, X (Euler angles)
            # Fix: Use rotate() instead of rotated()
            if theta_z != 0:
                shape = shape.rotate(Axis.Z, theta_z)
            if theta_y != 0:
                shape = shape.rotate(Axis.Y, theta_y)
            if theta_x != 0:
                shape = shape.rotate(Axis.X, theta_x)
        
        elif transform_type == TransformType.SCALE:
            factor_x = params.get('x_factor', 1.0)
            factor_y = params.get('y_factor', 1.0)
            factor_z = params.get('z_factor', 1.0)
            
            # If all factors are the same, use uniform scaling
            if factor_x == factor_y == factor_z:
                shape = shape.scaled(factor_x)
            else:
                # For non-uniform scaling, use specific scale factors
                shape = shape.scaled((factor_x, factor_y, factor_z))
        
        self.shape_stack.append(shape)
    
    def _apply_boolean(self, token):
        """Apply a boolean operation to the top two shapes on the stack"""
        if len(self.shape_stack) < 2:
            return
            
        boolean_type = token.get('boolean_type')
        
        shape2 = self.shape_stack.pop()
        shape1 = self.shape_stack.pop()
        
        if boolean_type == BooleanType.UNION:
            result = shape1 + shape2  # Union
        elif boolean_type == BooleanType.DIFFERENCE:
            result = shape1 - shape2  # Difference
        elif boolean_type == BooleanType.INTERSECTION:
            result = shape1 & shape2  # Intersection
        
        self.shape_stack.append(result)
    
    def get_final_shape(self):
        """Get the final shape (top of stack)"""
        if self.shape_stack:
            return self.shape_stack[-1]
        return None
    
    def to_pointcloud(self, n_points=2048):
        """Convert the final shape to a point cloud"""
        if not self.shape_stack:
            return None
            
        shape = self.get_final_shape()
        points = self._sample_points_from_shape(shape, n_points)
        
        return points
    
    def _sample_points_from_shape(self, shape, n_points):
        """Sample points from a shape using mesh export and Open3D"""
        # Export the shape to a temporary STL file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as temp_file:
            temp_filename = temp_file.name
            
        try:
            # Export the shape to STL
            export_stl(shape, temp_filename)
            
            # Use Open3D to read the STL file
            mesh = o3d.io.read_triangle_mesh(temp_filename)
            
            # Ensure the mesh has normals for proper sampling
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            # Sample points from the mesh surface
            pcd = mesh.sample_points_poisson_disk(n_points)
            
            # If Poisson disk sampling returns fewer points than requested,
            # use uniform sampling to fill the gap
            if len(pcd.points) < n_points:
                remaining_points = n_points - len(pcd.points)
                additional_pcd = mesh.sample_points_uniformly(remaining_points)
                pcd = pcd + additional_pcd
            
            # Extract points as numpy array
            points = np.asarray(pcd.points)
            
            # Normalize the points
            points = self._normalize_points(points)
            
            return points
            
        except Exception as e:
            print(f"Error sampling points from shape: {e}")
            # Fallback to the simpler method using bounding box
            return self._sample_points_fallback(shape, n_points)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    def _normalize_points(self, points):
        """Center and scale points to unit cube"""
        # Center
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale
        max_dist = np.max(np.abs(points))
        if max_dist > 0:  # Avoid division by zero
            points = points / max_dist
        
        return points
    
    def _sample_points_fallback(self, shape, n_points):
        """Fallback method to sample points using the bounding box approach"""
        # Get bounding box of the shape
        bbox = shape.bounding_box()
        
        # Extract min and max corners
        min_corner = (bbox.min.X, bbox.min.Y, bbox.min.Z)
        max_corner = (bbox.max.X, bbox.max.Y, bbox.max.Z)
        
        # Generate random points within bounding box
        points = []
        attempts = 0
        max_attempts = n_points * 100  # Limit the number of attempts
        
        while len(points) < n_points and attempts < max_attempts:
            attempts += 1
            
            # Generate random point in bounding box
            pt = (
                random.uniform(min_corner[0], max_corner[0]),
                random.uniform(min_corner[1], max_corner[1]),
                random.uniform(min_corner[2], max_corner[2])
            )
            
            # Check if point is inside the shape using the contains_point method
            try:
                if is_point_inside(shape, pt):
                    points.append(pt)
            except:
                # If the check fails (e.g., due to API limitations),
                # add the point anyway to ensure we get enough points
                if attempts > max_attempts * 0.8:  # In last 20% of attempts
                    points.append(pt)
        
        # If we couldn't get enough points, pad with random points
        while len(points) < n_points:
            pt = (
                random.uniform(min_corner[0], max_corner[0]),
                random.uniform(min_corner[1], max_corner[1]),
                random.uniform(min_corner[2], max_corner[2])
            )
            points.append(pt)
            
        points_array = np.array(points)
        
        # Normalize the points
        return self._normalize_points(points_array)


def is_point_inside(shape, point):
    """Check if a point is inside a shape using build123d's API"""
    # Convert point to tuple if it's not already
    if not isinstance(point, tuple):
        point = tuple(point)
    
    # Use build123d's contains_point method if available
    try:
        result = shape.contains_point(point)
        return result
    except:
        # Fallback for shapes that don't support contains_point
        try:
            # Alternative method if available in build123d
            return shape.is_inside(point)
        except:
            # Last resort: assume point is inside if we can't determine
            return True


def compute_chamfer_distance(points1, points2):
    """
    Compute the Chamfer distance between two point clouds.
    This is a common metric for comparing 3D shapes.
    """
    # Create Open3D point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    
    # Compute distances from points1 to points2
    distances1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    
    # Compute distances from points2 to points1
    distances2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    
    # Chamfer distance is the sum of mean distances in both directions
    chamfer_dist = np.mean(distances1) + np.mean(distances2)
    
    return chamfer_dist

def generate_shape_from_tokens(tokens):
    """
    Generate a shape from a list of operation tokens
    and return it as a point cloud
    """
    generator = ShapeGenerator()
    
    for token in tokens:
        generator.execute_token(token)
        
    points = generator.to_pointcloud()
    return points

def compute_shape_loss(original_points, tokens):
    """
    Compute the loss between an original shape and a shape
    generated from tokens
    """
    generated_points = generate_shape_from_tokens(tokens)
    
    if generated_points is None:
        return float('inf')  # Return high loss if generation failed
        
    return compute_chamfer_distance(original_points, generated_points)
