# Geometrize

A deep learning system for reconstructing 3D shapes using primitive shape operations.

## Overview

Geometrize is a machine learning project that learns to represent complex 3D shapes as a sequence of primitive shapes and transformations. Given an STL file of a 3D shape, the system learns the minimal set of operations needed to recreate it.

The system uses a Variational Autoencoder (VAE) architecture with:
- PointNet encoder to process point cloud data
- Transformer decoder to generate sequential shape operations
- Discrete binning for numerical parameter prediction

## Features

- Point cloud processing and normalization from STL files
- Primitive shape generation (sphere, cube, cylinder, etc.)
- Boolean operations (union, difference, etc.)
- Transformations (translate, rotate, scale)
- Variational autoencoder for shape representation learning

## Installation

```bash
# Clone the repository
git clone https://github.com/robh96/geometrize.git
cd geometrize

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Test Shapes

```python
from geometrize.generate_test_shapes import save_test_shapes

# Generate and save basic test shapes (sphere, cube, cylinder)
save_test_shapes()
```

### Training the Model

```python
from geometrize.pipeline import train

# Train the model
encoder, decoder, train_losses, val_losses = train()
```

### Converting Point Clouds to Shape Operations

```python
import numpy as np
from geometrize.model import PointNetEncoder
from geometrize.pipeline import reparameterize, convert_tokens_to_operations
from geometrize.shape_generator import generate_shape_from_tokens

# Load your model
encoder = PointNetEncoder(latent_dim=256)
# Load weights...

# Process point cloud
point_cloud = np.load('your_point_cloud.npy')
point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float).unsqueeze(0)

# Encode and generate tokens
with torch.no_grad():
    mu, logvar = encoder(point_cloud_tensor)
    z = reparameterize(mu, logvar)
    token_probs, bin_probs, _ = decoder(z)
    
    pred_tokens = torch.argmax(token_probs, dim=-1).cpu().numpy()[0]
    pred_bins = torch.argmax(bin_probs, dim=-1).cpu().numpy()[0]
    
    # Convert to operations
    operations = convert_tokens_to_operations(pred_tokens, pred_bins)
    
    # Generate shape from operations
    output_points = generate_shape_from_tokens(operations)
```

## Testing

Run the test scripts to verify functionality:

```bash
python -m geometrize.test_model
python -m geometrize.test_losses
python -m geometrize.test_pipeline
python -m geometrize.test_shape_generator
```

## Project Structure

- `.data/`: Sample STL files for testing
- `.checkpoints/`: Model checkpoints saved during training
- `.outputs/`: Output files, visualizations, and logs
- `geometrize/`: Main package
  - `config.py`: Configuration settings
  - `data_processing.py`: Point cloud processing utilities
  - `generate_test_shapes.py`: Utilities to create test shapes
  - `losses.py`: Loss function implementations
  - `model.py`: Neural network model definitions
  - `pipeline.py`: Training and inference pipeline
  - `shape_generator.py`: 3D shape generation from operations
  - `token_map.py`: Mapping between tokens and operations
  - Various test files to validate different components

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Build123d for CAD operations
- PyTorch for deep learning framework
- Open3D for point cloud processing
