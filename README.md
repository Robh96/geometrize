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

### Command Line Usage

Geometrize can also be used directly from the command line for both training and inference.

#### Training

To train a new model:

```bash
# Basic training with default parameters
python -m geometrize.train

# Training with custom parameters
python -m geometrize.train --batch-size 32 --epochs 100 --learning-rate 0.0005 --beta 0.1 --shape-weight 0.5
```

Options:
- `--batch-size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--learning-rate`: Learning rate for the optimizer (default: 0.001)
- `--beta`: Beta value for the KL divergence term (default: 0.1)
- `--shape-weight`: Weight for the shape loss component (default: 0.5)
- `--data-dir`: Directory containing STL files for training (default: ./data)
- `--output-dir`: Directory to save outputs and visualizations (default: ./outputs)
- `--checkpoint-dir`: Directory to save model checkpoints (default: ./checkpoints)
- `--resume`: Path to checkpoint for resuming training (optional)

#### Inference

To run inference on STL files:

```bash
# Run inference on a single STL file
python -m geometrize.infer --input path/to/shape.stl --model path/to/model.pth --output path/to/output.stl

# Run inference on all STL files in a directory
python -m geometrize.infer --input-dir path/to/stl_dir --model path/to/model.pth --output-dir path/to/output_dir
```

Options:
- `--input`: Path to input STL file (for single file inference)
- `--input-dir`: Directory containing STL files (for batch inference)
- `--model`: Path to the trained model checkpoint
- `--output`: Path to save the reconstructed STL file (for single file inference)
- `--output-dir`: Directory to save reconstructed STL files (for batch inference)
- `--visualize`: Flag to visualize comparisons between original and reconstructed shapes
- `--num-points`: Number of points to sample from each STL file (default: 2048)
- `--latent-dim`: Latent dimension size matching the trained model (default: 256)

Example:
```bash
# Process a single file with visualization
python -m geometrize.infer --input myshape.stl --model best_model.pth --output reconstructed.stl --visualize

# Process all files in a directory
python -m geometrize.infer --input-dir ./test_shapes --model ./checkpoints/final_model.pth --output-dir ./reconstructions
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

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Build123d for CAD operations
- PyTorch for deep learning framework
- Open3D for point cloud processing
