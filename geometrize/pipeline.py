import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import time
import random
import matplotlib.pyplot as plt
from geometrize.model import PointNetEncoder, TransformerDecoder
from geometrize.losses import vae_loss, hausdorff_distance
from geometrize.data_processing import load_stl_to_pointcloud
from geometrize.shape_generator import (
    generate_shape_from_tokens, compute_shape_loss,
    OperationType, PrimitiveType, TransformType, BooleanType  # Add these imports at the module level
)
from geometrize.config import Config
from geometrize.token_map import TokenMap
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

class ShapeDataset(Dataset):
    def __init__(self, data_paths, token_sequences):
        self.data_paths = data_paths
        self.token_sequences = token_sequences
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        try:
            # Load and process point cloud
            point_cloud = self._load_and_process_pointcloud(self.data_paths[idx])
            
            # Pad token sequences to fixed length
            tokens = self._pad_token_sequence(self.token_sequences[idx])
            
            return torch.tensor(point_cloud, dtype=torch.float), tokens
        except Exception as e:
            # If there's any error, return fallback data
            logging.error(f"Error loading sample {idx} from {self.data_paths[idx]}: {e}")
            fallback_points = np.zeros((Config.num_points, 3))
            fallback_tokens = self._create_fallback_tokens()
            return torch.tensor(fallback_points, dtype=torch.float), fallback_tokens
    
    def _load_and_process_pointcloud(self, file_path):
        """Load and process point cloud with robust error handling"""
        try:
            # First try the standard loading
            point_cloud = load_stl_to_pointcloud(file_path, Config.num_points)
            
            # Verify point count - if wrong, use simpler approach
            if len(point_cloud) != Config.num_points:
                logging.warning(
                    f"Point cloud from {file_path} has {len(point_cloud)} points "
                    f"instead of the expected {Config.num_points}. Using simple sampling."
                )
                # Use simpler, more robust resampling
                point_cloud = self._simple_resample_points(point_cloud, Config.num_points)
                
        except Exception as e:
            logging.warning(f"Error loading point cloud from {file_path}: {e}")
            # Create a simple shape (sphere) as fallback
            point_cloud = self._create_fallback_sphere(Config.num_points)
            
        return point_cloud
    
    def _simple_resample_points(self, points, n_points):
        """Simple but robust point resampling that always works"""
        if len(points) == 0:
            # Handle empty point cloud
            return self._create_fallback_sphere(n_points)
            
        if len(points) >= n_points:
            # Downsample: random sampling without replacement
            indices = np.random.choice(len(points), n_points, replace=False)
            return points[indices]
        else:
            # Upsample: random sampling with replacement for remaining points
            original_points = points.copy()
            indices = np.random.choice(len(points), n_points - len(points), replace=True)
            extra_points = points[indices]
            # Add small random jitter to avoid exact duplicates
            extra_points += np.random.normal(0, 0.01, extra_points.shape)
            return np.vstack([original_points, extra_points])
    
    def _create_fallback_sphere(self, n_points):
        """Create a simple unit sphere point cloud for fallback"""
        # Generate points on unit sphere using Fibonacci sphere algorithm
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        points = np.column_stack((x, y, z))
        return points
    
    def _pad_token_sequence(self, sequence):
        """Pad or truncate token sequence to fixed length"""
        # Get current sequence length
        seq_len = sequence.size(0)
        max_len = Config.max_seq_length
        
        # If sequence is already the right length, return as is
        if seq_len == max_len:
            return sequence
        
        # Get the token map for NULL token
        token_map = TokenMap()
        null_token_id = token_map.get_id('NULL')
        
        # If sequence is too long, truncate
        if seq_len > max_len:
            return sequence[:max_len]
        
        # If sequence is too short, pad with NULL tokens
        padding_len = max_len - seq_len
        padding = torch.zeros(padding_len, 2, dtype=torch.long)
        padding[:, 0] = null_token_id  # Set first column (token) to NULL
        # Second column (bin) stays as 0
        
        # Concatenate original sequence with padding
        padded_sequence = torch.cat([sequence, padding], dim=0)
        
        return padded_sequence
    
    def _create_fallback_tokens(self):
        """Create a fallback token sequence"""
        token_map = TokenMap()
        null_token_id = token_map.get_id('NULL')
        
        # Create a sequence of NULL tokens
        fallback = torch.zeros((Config.max_seq_length, 2), dtype=torch.long)
        fallback[:, 0] = null_token_id  # Set first column (token) to NULL
        # Second column (bins) stays as 0
        
        return fallback

def reparameterize(mu, logvar):
    """Reparameterization trick for VAE"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z

def save_loss_plot(train_losses, val_losses, filename):
    """Save a plot of training and validation losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def save_reconstruction_examples(epoch, encoder, decoder, test_dataloader, device):
    """Save examples of reconstructed shapes"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(Config.output_dir, "reconstructions"), exist_ok=True)
    
    # Get a batch of data
    point_clouds, token_seqs = next(iter(test_dataloader))
    point_clouds = point_clouds.to(device)
    
    # Encode and decode
    mu, logvar = encoder(point_clouds[:4])  # Only process a few examples
    z = reparameterize(mu, logvar)
    token_probs, bin_probs, _ = decoder(z)
    
    # Get predicted tokens
    pred_tokens = torch.argmax(token_probs, dim=-1).cpu().numpy()
    pred_bins = torch.argmax(bin_probs, dim=-1).cpu().numpy()
    
    # Create and save figures
    for i in range(min(4, len(point_clouds))):
        # Convert tokens to operations and generate shape
        token_ops = convert_tokens_to_operations(pred_tokens[i], pred_bins[i])
        try:
            # Generate point cloud from predicted tokens
            gen_points = generate_shape_from_tokens(token_ops)
            
            if gen_points is not None:
                # Plot original and reconstructed point cloud side by side
                fig = plt.figure(figsize=(12, 6))
                
                # Original point cloud
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(
                    point_clouds[i, :, 0].cpu().numpy(),
                    point_clouds[i, :, 1].cpu().numpy(),
                    point_clouds[i, :, 2].cpu().numpy(),
                    c=point_clouds[i, :, 2].cpu().numpy(),
                    cmap='viridis',
                    s=10
                )
                ax1.set_title('Original')
                
                # Generated point cloud
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(
                    gen_points[:, 0],
                    gen_points[:, 1],
                    gen_points[:, 2],
                    c=gen_points[:, 2],
                    cmap='viridis',
                    s=10
                )
                ax2.set_title('Reconstructed')
                
                plt.savefig(os.path.join(Config.output_dir, f"reconstructions/epoch_{epoch}_sample_{i}.png"))
                plt.close()
        except Exception as e:
            logging.error(f"Error generating reconstruction for sample {i}: {e}")

def convert_tokens_to_operations(tokens, bins, token_map=None):
    """Convert token IDs and bin values to operation dictionaries"""
    if token_map is None:
        token_map = TokenMap()
    
    operations = []
    i = 0
    
    while i < len(tokens) and tokens[i] != token_map.get_id('NULL'):
        token_name = token_map.get_token(int(tokens[i]))
        
        if token_name == 'SPHERE':
            # Format: SPHERE, CENTER_X, bin, CENTER_Y, bin, CENTER_Z, bin, RADIUS, bin
            if i + 8 < len(tokens):
                operation = {
                    'operation': OperationType.PRIMITIVE,
                    'primitive_type': PrimitiveType.SPHERE,
                    'parameters': {
                        'center_x': (bins[i+2] / Config.num_bins) * 2 - 1,
                        'center_y': (bins[i+4] / Config.num_bins) * 2 - 1,
                        'center_z': (bins[i+6] / Config.num_bins) * 2 - 1,
                        'radius': (bins[i+8] / Config.num_bins) * 0.5  # Scale to reasonable size
                    }
                }
                operations.append(operation)
                i += 9
            else:
                i += 1
                
        elif token_name == 'CUBOID':
            # Format: CUBOID, CENTER_X, bin, CENTER_Y, bin, CENTER_Z, bin, DX, bin, DY, bin, DZ, bin
            if i + 12 < len(tokens):
                operation = {
                    'operation': OperationType.PRIMITIVE,
                    'primitive_type': PrimitiveType.CUBE,
                    'parameters': {
                        'center_x': (bins[i+2] / Config.num_bins) * 2 - 1,
                        'center_y': (bins[i+4] / Config.num_bins) * 2 - 1,
                        'center_z': (bins[i+6] / Config.num_bins) * 2 - 1,
                        'size_x': (bins[i+8] / Config.num_bins),
                        'size_y': (bins[i+10] / Config.num_bins),
                        'size_z': (bins[i+12] / Config.num_bins)
                    }
                }
                operations.append(operation)
                i += 13
            else:
                i += 1
                
        elif token_name == 'UNION' or token_name == 'SUBTRACT':
            # Format: UNION/SUBTRACT, OBJECT_1, bin, OBJECT_2, bin
            if i + 4 < len(tokens):
                bool_type = BooleanType.UNION if token_name == 'UNION' else BooleanType.DIFFERENCE
                operation = {
                    'operation': OperationType.BOOLEAN,
                    'boolean_type': bool_type
                }
                operations.append(operation)
                i += 5
            else:
                i += 1
                
        elif token_name == 'TRANSLATE':
            # Format: TRANSLATE, OBJECT, bin, DX, bin, DY, bin, DZ, bin
            if i + 8 < len(tokens):
                operation = {
                    'operation': OperationType.TRANSFORM,
                    'transform_type': TransformType.TRANSLATE,
                    'parameters': {
                        'x': (bins[i+4] / Config.num_bins) * 2 - 1,
                        'y': (bins[i+6] / Config.num_bins) * 2 - 1,
                        'z': (bins[i+8] / Config.num_bins) * 2 - 1
                    }
                }
                operations.append(operation)
                i += 9
            else:
                i += 1
                
        elif token_name == 'ROTATE':
            # Format: ROTATE, OBJECT, bin, THETA_X, bin, THETA_Y, bin, THETA_Z, bin
            if i + 8 < len(tokens):
                operation = {
                    'operation': OperationType.TRANSFORM,
                    'transform_type': TransformType.ROTATE,
                    'parameters': {
                        'x': (bins[i+4] / Config.num_bins) * 360,  # Scale to degrees
                        'y': (bins[i+6] / Config.num_bins) * 360,
                        'z': (bins[i+8] / Config.num_bins) * 360
                    }
                }
                operations.append(operation)
                i += 9
            else:
                i += 1
                
        elif token_name == 'SCALE':
            # Format: SCALE, OBJECT, bin, SCALE_X, bin, SCALE_Y, bin, SCALE_Z, bin
            if i + 8 < len(tokens):
                operation = {
                    'operation': OperationType.TRANSFORM,
                    'transform_type': TransformType.SCALE,
                    'parameters': {
                        'x_factor': (bins[i+4] / Config.num_bins) * 2,  # Scale from 0 to 2
                        'y_factor': (bins[i+6] / Config.num_bins) * 2,
                        'z_factor': (bins[i+8] / Config.num_bins) * 2
                    }
                }
                operations.append(operation)
                i += 9
            else:
                i += 1
        else:
            # Skip unknown tokens
            i += 1
            
    return operations

def load_or_generate_data():
    """Load real data or generate synthetic data for training"""
    # Check if we have real data
    stl_files = glob.glob(os.path.join(Config.data_dir, "*.stl"))
    
    if len(stl_files) > 10:  # Arbitrary threshold for "enough" data
        logging.info(f"Found {len(stl_files)} STL files for training")
        # TODO: Load corresponding token sequences
        # For now, we'll generate synthetic token sequences
        token_sequences = [generate_synthetic_tokens() for _ in stl_files]
        return stl_files, token_sequences
    else:
        logging.info("Not enough real data found, generating synthetic data")
        # Generate synthetic data
        from geometrize.generate_test_shapes import save_test_shapes
        save_test_shapes()  # This will create basic shapes in the data directory
        
        # Get the newly generated shapes
        stl_files = glob.glob(os.path.join(Config.data_dir, "*.stl"))
        token_sequences = [generate_synthetic_tokens() for _ in stl_files]
        return stl_files, token_sequences

def generate_synthetic_tokens():
    """Generate synthetic token sequences for testing"""
    # This is a placeholder. In a real implementation, we'd have actual token sequences.
    token_map = TokenMap()
    seq_length = random.randint(5, 30)  # Random sequence length
    
    # Generate random token and bin indices
    tokens = torch.randint(0, len(token_map.token_to_id), (seq_length,))
    bins = torch.randint(0, Config.num_bins, (seq_length,))
    
    # Stack tokens and bins to create sequence
    sequence = torch.stack([tokens, bins], dim=1)
    
    return sequence

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get data for training
    data_paths, token_sequences = load_or_generate_data()
    
    # Split data into train and validation sets
    train_ratio = 0.8
    train_size = int(len(data_paths) * train_ratio)
    
    train_paths = data_paths[:train_size]
    train_tokens = token_sequences[:train_size]
    
    val_paths = data_paths[train_size:]
    val_tokens = token_sequences[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = ShapeDataset(train_paths, train_tokens)
    val_dataset = ShapeDataset(val_paths, val_tokens)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4
    )
    
    # Initialize models
    encoder = PointNetEncoder(latent_dim=Config.latent_dim).to(device)
    decoder = TransformerDecoder(
        latent_dim=Config.latent_dim,
        token_dim=len(TokenMap().token_to_id),
        num_bins=Config.num_bins
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=Config.learning_rate
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create directories for checkpoints and outputs
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Training metrics
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_token_losses = []
    train_bin_losses = []
    train_kl_losses = []
    train_shape_losses = []
    
    # Training loop
    for epoch in range(Config.num_epochs):
        # Training phase
        encoder.train()
        decoder.train()
        
        train_loss = 0
        train_token_loss = 0
        train_bin_loss = 0
        train_kl_loss = 0
        train_shape_loss = 0
        
        start_time = time.time()
        
        for batch_idx, (point_clouds, token_seqs) in enumerate(train_dataloader):
            # Move data to device
            point_clouds = point_clouds.to(device)
            token_seqs = token_seqs.to(device)
            
            # Extract token and bin values
            target_tokens = token_seqs[:, :, 0]
            target_bins = token_seqs[:, :, 1]
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Encode point clouds
            mu, logvar = encoder(point_clouds)
            
            # Sample latent vector using reparameterization trick
            z = reparameterize(mu, logvar)
            
            # Decode latent vector to token and bin predictions
            token_probs, bin_probs, offsets = decoder(z, token_seqs)
            
            # Standard VAE loss calculation
            loss, (token_loss, bin_loss, kl_loss, shape_loss) = vae_loss(
                token_probs, bin_probs, target_tokens, target_bins, mu, logvar
            )
            
            # For some batches, calculate shape loss
            if batch_idx % 5 == 0:  # Every 5 batches to save computation
                try:
                    # Get predicted tokens and bins
                    pred_tokens = torch.argmax(token_probs, dim=-1)
                    pred_bins = torch.argmax(bin_probs, dim=-1)
                    
                    # Sample a few examples from the batch
                    sample_indices = torch.randint(0, point_clouds.size(0), (min(2, point_clouds.size(0)),))
                    
                    for idx in sample_indices:
                        # Convert predicted tokens to operations
                        token_ops = convert_tokens_to_operations(
                            pred_tokens[idx].cpu().numpy(),
                            pred_bins[idx].cpu().numpy()
                        )
                        
                        # Generate point cloud from operations
                        gen_points = generate_shape_from_tokens(token_ops)
                        
                        if gen_points is not None:
                            # Convert to tensor and prepare for distance calculation
                            gen_points_tensor = torch.tensor(gen_points, 
                                                            dtype=torch.float, 
                                                            device=device).unsqueeze(0)
                            orig_points = point_clouds[idx].unsqueeze(0)
                            
                            # Calculate Hausdorff distance
                            shape_distance = hausdorff_distance(orig_points, gen_points_tensor)
                            shape_loss = shape_distance.mean()
                            
                            # Add weighted shape loss to the total loss
                            shape_weight = Config.shape_weight if hasattr(Config, 'shape_weight') else 0.1
                            loss = loss + shape_weight * shape_loss
                            break  # Just use one example to save computation
                except Exception as e:
                    logging.warning(f"Error calculating shape loss: {e}")
            
            # Backpropagation
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Track component losses
            train_loss += loss.item()
            train_token_loss += token_loss.item()
            train_bin_loss += bin_loss.item()
            train_kl_loss += kl_loss.item()
            if hasattr(shape_loss, 'item'):
                train_shape_loss += shape_loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{Config.num_epochs} | Batch {batch_idx}/{len(train_dataloader)} | "
                           f"Loss: {loss.item():.4f} | Shape Loss: {shape_loss.item() if hasattr(shape_loss, 'item') else 0:.4f}")
        
        # Calculate average training losses
        avg_train_loss = train_loss / len(train_dataloader)
        avg_token_loss = train_token_loss / len(train_dataloader)
        avg_bin_loss = train_bin_loss / len(train_dataloader)
        avg_kl_loss = train_kl_loss / len(train_dataloader)
        avg_shape_loss = train_shape_loss / len(train_dataloader) if train_shape_loss > 0 else 0
        
        train_losses.append(avg_train_loss)
        train_token_losses.append(avg_token_loss)
        train_bin_losses.append(avg_bin_loss)
        train_kl_losses.append(avg_kl_loss)
        train_shape_losses.append(avg_shape_loss)
        
        # Validation phase
        encoder.eval()
        decoder.eval()
        
        val_loss = 0
        
        with torch.no_grad():
            for point_clouds, token_seqs in val_dataloader:
                # Move data to device
                point_clouds = point_clouds.to(device)
                token_seqs = token_seqs.to(device)
                
                # Extract token and bin values
                target_tokens = token_seqs[:, :, 0]
                target_bins = token_seqs[:, :, 1]
                
                # Encode and decode
                mu, logvar = encoder(point_clouds)
                z = reparameterize(mu, logvar)
                token_probs, bin_probs, offsets = decoder(z, token_seqs)
                
                # Compute loss
                loss = vae_loss(token_probs, bin_probs, target_tokens, target_bins, mu, logvar)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, os.path.join(Config.checkpoint_dir, 'best_model.pth'))
            logging.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, os.path.join(Config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Log epoch