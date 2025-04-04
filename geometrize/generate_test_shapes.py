import os
import numpy as np
from stl import mesh
from geometrize.config import Config

def generate_sphere(radius=1.0, resolution=36):
    """Generate a sphere mesh."""
    # Create sphere using spherical coordinates
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create mesh
    vertices = []
    faces = []
    
    for i in range(resolution-1):
        for j in range(resolution-1):
            v0 = i * resolution + j
            v1 = i * resolution + (j + 1)
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + (j + 1)
            
            # First triangle
            vertices.append([x.flatten()[v0], y.flatten()[v0], z.flatten()[v0]])
            vertices.append([x.flatten()[v1], y.flatten()[v1], z.flatten()[v1]])
            vertices.append([x.flatten()[v2], y.flatten()[v2], z.flatten()[v2]])
            faces.append([len(vertices)-3, len(vertices)-2, len(vertices)-1])
            
            # Second triangle
            vertices.append([x.flatten()[v1], y.flatten()[v1], z.flatten()[v1]])
            vertices.append([x.flatten()[v3], y.flatten()[v3], z.flatten()[v3]])
            vertices.append([x.flatten()[v2], y.flatten()[v2], z.flatten()[v2]])
            faces.append([len(vertices)-3, len(vertices)-2, len(vertices)-1])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create the stl mesh
    sphere = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            sphere.vectors[i][j] = vertices[face[j]]
    
    return sphere

def generate_cube(size=1.0):
    """Generate a cube mesh."""
    # Define the 8 vertices of the cube
    vertices = np.array([
        [-size, -size, -size],
        [+size, -size, -size],
        [+size, +size, -size],
        [-size, +size, -size],
        [-size, -size, +size],
        [+size, -size, +size],
        [+size, +size, +size],
        [-size, +size, +size]
    ])
    
    # Define the 12 triangles composing the cube
    faces = np.array([
        [0, 3, 1], [1, 3, 2],  # bottom face
        [0, 4, 7], [0, 7, 3],  # left face
        [4, 5, 6], [4, 6, 7],  # top face
        [5, 1, 2], [5, 2, 6],  # right face
        [2, 3, 6], [3, 7, 6],  # front face
        [0, 1, 5], [0, 5, 4]   # back face
    ])
    
    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[face[j]]
    
    return cube

def generate_cylinder(radius=0.5, height=2.0, resolution=36):
    """Generate a cylinder mesh."""
    # Create cylinder using cylindrical coordinates
    theta = np.linspace(0, 2 * np.pi, resolution)
    
    # Create top and bottom circles
    bottom_circle_x = radius * np.cos(theta)
    bottom_circle_y = radius * np.sin(theta)
    bottom_circle_z = np.ones_like(theta) * -height/2
    
    top_circle_x = radius * np.cos(theta)
    top_circle_y = radius * np.sin(theta)
    top_circle_z = np.ones_like(theta) * height/2
    
    # Create mesh
    vertices = []
    faces = []
    
    # Add side faces
    for i in range(resolution):
        v0 = i
        v1 = (i + 1) % resolution
        v2 = i + resolution
        v3 = (i + 1) % resolution + resolution
        
        # vertices for the side rectangles
        vertices.append([bottom_circle_x[v0], bottom_circle_y[v0], bottom_circle_z[v0]])
        vertices.append([bottom_circle_x[v1], bottom_circle_y[v1], bottom_circle_z[v1]])
        vertices.append([top_circle_x[v0], top_circle_y[v0], top_circle_z[v0]])
        vertices.append([top_circle_x[v1], top_circle_y[v1], top_circle_z[v1]])
        
        # Two triangles to make a rectangle
        faces.append([len(vertices)-4, len(vertices)-3, len(vertices)-2])  # First triangle
        faces.append([len(vertices)-3, len(vertices)-1, len(vertices)-2])  # Second triangle
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create the stl mesh
    cylinder = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cylinder.vectors[i][j] = vertices[face[j]]
    
    return cylinder

def save_test_shapes():
    """Generate and save test shapes as STL files."""
    # Ensure data directory exists
    os.makedirs(Config.data_dir, exist_ok=True)
    
    # Generate and save shapes
    sphere = generate_sphere()
    cube = generate_cube()
    cylinder = generate_cylinder()
    
    sphere.save(os.path.join(Config.data_dir, 'sphere.stl'))
    cube.save(os.path.join(Config.data_dir, 'cube.stl'))
    cylinder.save(os.path.join(Config.data_dir, 'cylinder.stl'))
    
    print(f"Test shapes saved to {Config.data_dir}")

if __name__ == "__main__":
    save_test_shapes()