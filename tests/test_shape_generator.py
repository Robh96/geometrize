import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from geometrize.shape_generator import ShapeGenerator, OperationType, PrimitiveType, TransformType, BooleanType
from build123d import *
import os
from geometrize.config import Config

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
    o3d.visualization.draw_geometries([pcd, frame], window_name=title)
    
    return pcd

def test_primitive_creation():
    """Test creation of various primitives."""
    print("\n=== Testing Primitive Creation ===")
    generator = ShapeGenerator()
    
    # Create a sphere
    sphere_token = {
        'operation': OperationType.PRIMITIVE,
        'primitive_type': PrimitiveType.SPHERE,
        'parameters': {'radius': 0.5}
    }
    generator.execute_token(sphere_token)
    
    # Get the point cloud
    points = generator.to_pointcloud(n_points=1024)
    assert points is not None, "Points should not be None"
    assert points.shape[0] > 0, "Should have generated some points"
    assert points.shape[1] == 3, "Points should be 3D coordinates"
    
    print("Sphere created successfully")
    test_primitive_creation.points = points

def test_transformations():
    """Test various transformations."""
    print("\n=== Testing Transformations ===")
    generator = ShapeGenerator()
    
    # Create a box
    box_token = {
        'operation': OperationType.PRIMITIVE,
        'primitive_type': PrimitiveType.CUBE,
        'parameters': {'size': 0.8}
    }
    generator.execute_token(box_token)
    
    # Translate the box
    translate_token = {
        'operation': OperationType.TRANSFORM,
        'transform_type': TransformType.TRANSLATE,
        'parameters': {'x': 0.5, 'y': 0.2, 'z': -0.3}
    }
    generator.execute_token(translate_token)
    
    # Rotate the box
    rotate_token = {
        'operation': OperationType.TRANSFORM,
        'transform_type': TransformType.ROTATE,
        'parameters': {'x': 30, 'y': 15, 'z': 45}
    }
    generator.execute_token(rotate_token)
    
    # Get the point cloud
    points = generator.to_pointcloud(n_points=1024)
    
    # Verify the points
    assert points is not None, "Points should not be None"
    assert points.shape[0] > 0, "Should have generated some points"
    assert points.shape[1] == 3, "Points should be 3D coordinates"
    
    print("Box with transformations created successfully")
    
    # Store for other tests without returning
    test_transformations.points = points

def test_boolean_operations():
    """Test boolean operations."""
    print("\n=== Testing Boolean Operations ===")
    generator = ShapeGenerator()
    
    # Create a sphere
    sphere_token = {
        'operation': OperationType.PRIMITIVE,
        'primitive_type': PrimitiveType.SPHERE,
        'parameters': {'radius': 0.6}
    }
    generator.execute_token(sphere_token)
    
    # Create a box
    box_token = {
        'operation': OperationType.PRIMITIVE,
        'primitive_type': PrimitiveType.CUBE,
        'parameters': {'size': 0.7}
    }
    generator.execute_token(box_token)
    
    # Perform a boolean operation
    boolean_token = {
        'operation': OperationType.BOOLEAN,
        'boolean_type': BooleanType.DIFFERENCE
    }
    generator.execute_token(boolean_token)
    
    # Get the point cloud
    points = generator.to_pointcloud(n_points=1024)
    
    # Verify the points
    assert points is not None, "Points should not be None"
    assert points.shape[0] > 0, "Should have generated some points"
    assert points.shape[1] == 3, "Points should be 3D coordinates"
    
    print("Boolean operation completed successfully")
    
    # Store for other tests without returning
    test_boolean_operations.points = points

def test_complex_shape():
    """Test creating a more complex shape with multiple operations."""
    print("\n=== Testing Complex Shape Creation ===")
    generator = ShapeGenerator()
    
    # Create a cylinder
    cylinder_token = {
        'operation': OperationType.PRIMITIVE,
        'primitive_type': PrimitiveType.CYLINDER,
        'parameters': {'radius': 0.4, 'height': 1.5}
    }
    generator.execute_token(cylinder_token)
    
    # Create a sphere
    sphere_token = {
        'operation': OperationType.PRIMITIVE,
        'primitive_type': PrimitiveType.SPHERE,
        'parameters': {'radius': 0.5}
    }
    generator.execute_token(sphere_token)
    
    # Translate the sphere
    translate_token = {
        'operation': OperationType.TRANSFORM,
        'transform_type': TransformType.TRANSLATE,
        'parameters': {'x': 0, 'y': 0, 'z': 0.8}
    }
    generator.execute_token(translate_token)
    
    # Union them
    union_token = {
        'operation': OperationType.BOOLEAN,
        'boolean_type': BooleanType.UNION
    }
    generator.execute_token(union_token)
    
    # Create another sphere for subtraction
    sphere2_token = {
        'operation': OperationType.PRIMITIVE,
        'primitive_type': PrimitiveType.SPHERE,
        'parameters': {'radius': 0.2}
    }
    generator.execute_token(sphere2_token)
    
    # Translate the second sphere
    translate2_token = {
        'operation': OperationType.TRANSFORM,
        'transform_type': TransformType.TRANSLATE,
        'parameters': {'x': 0.5, 'y': 0, 'z': 0.5}
    }
    generator.execute_token(translate2_token)
    
    # Subtract
    diff_token = {
        'operation': OperationType.BOOLEAN,
        'boolean_type': BooleanType.DIFFERENCE
    }
    generator.execute_token(diff_token)
    
    # Get the point cloud
    points = generator.to_pointcloud(n_points=2048)
    
    # Verify the points
    assert points is not None, "Points should not be None"
    assert points.shape[0] > 0, "Should have generated some points"
    assert points.shape[1] == 3, "Points should be 3D coordinates"
    
    print("Complex shape created successfully")
    
    # Store for other tests without returning
    test_complex_shape.points = points

def test_generate_shape_from_tokens():
    """Test the helper function to generate shapes from tokens."""
    from geometrize.shape_generator import generate_shape_from_tokens, compute_chamfer_distance
    
    # Define token sequence
    tokens = [
        {
            'operation': OperationType.PRIMITIVE,
            'primitive_type': PrimitiveType.CYLINDER,
            'parameters': {'radius': 0.3, 'height': 1.0}
        },
        {
            'operation': OperationType.TRANSFORM,
            'transform_type': TransformType.ROTATE,
            'parameters': {'x': 0, 'y': 90, 'z': 0}
        }
    ]
    
    # Generate point cloud
    points = generate_shape_from_tokens(tokens)
    print("\n=== Testing generate_shape_from_tokens ===")
    print(f"Generated point cloud with {len(points)} points")
    
    # Verify the points
    assert points is not None, "Points should not be None"
    assert points.shape[0] > 0, "Should have generated some points"
    assert points.shape[1] == 3, "Points should be 3D coordinates"
    
    # Create a similar shape with slight differences
    modified_tokens = tokens.copy()
    modified_tokens[0]['parameters']['radius'] = 0.35  # Slightly different radius
    points2 = generate_shape_from_tokens(modified_tokens)
    
    # Check second set of points
    assert points2 is not None, "Second points should not be None"
    assert points2.shape[0] > 0, "Should have generated some points"
    assert points2.shape[1] == 3, "Points should be 3D coordinates"
    
    # Compute chamfer distance
    distance = compute_chamfer_distance(points, points2)
    print(f"Chamfer distance between similar shapes: {distance:.6f}")
    
    # Store for other tests without returning
    test_generate_shape_from_tokens.points = points
    test_generate_shape_from_tokens.points2 = points2

def save_visualization(points, filename):
    """Save point cloud visualization as an image."""
    os.makedirs(Config.output_dir, exist_ok=True)
    filepath = os.path.join(Config.output_dir, filename)
    
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
    ax.set_title(f"Point Cloud Visualization: {filename}")
    
    # Save figure
    plt.savefig(filepath)
    plt.close()
    print(f"Visualization saved to {filepath}")

def run_all_tests():
    """Run all tests and visualize results."""
    # Make sure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Test primitive creation
    test_primitive_creation()
    save_visualization(test_primitive_creation.points, "sphere_primitive.png")
    
    # Test transformations
    test_transformations()
    save_visualization(test_transformations.points, "transformed_box.png")
    
    # Test boolean operations
    test_boolean_operations()
    save_visualization(test_boolean_operations.points, "boolean_difference.png")
    
    # Test complex shape
    test_complex_shape()
    save_visualization(test_complex_shape.points, "complex_shape.png")
    
    # Test generate from tokens function
    test_generate_shape_from_tokens()
    save_visualization(test_generate_shape_from_tokens.points, "generated_from_tokens.png")
    
    print("\nAll tests completed! Visualizations saved to output directory.")
    
    # Choose one shape to visualize with open3d
    print("\nDisplaying complex shape with Open3D viewer (close window to continue):")
    visualize_point_cloud(test_complex_shape.points, "Complex Shape Visualization")

if __name__ == "__main__":
    run_all_tests()
