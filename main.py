"""
Main entry point for Geometrize

This script provides a command-line interface to the main functionalities:
- Generate test shapes
- Run training
- Test components
- Perform inference
"""

import os
import argparse
import torch
from geometrize.config import Config
from geometrize.generate_test_shapes import save_test_shapes
from geometrize.pipeline import train, test

def setup_argparse():
    parser = argparse.ArgumentParser(description='Geometrize - 3D Shape Representation Learning')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate test shapes command
    generate_parser = subparsers.add_parser('generate', help='Generate test shapes')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=Config.num_epochs, 
                             help='Number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=Config.batch_size,
                             help='Batch size for training')
    train_parser.add_argument('--learning-rate', type=float, default=Config.learning_rate,
                             help='Learning rate for training')
    train_parser.add_argument('--beta', type=float, default=Config.beta,
                             help='Beta parameter for VAE')
    train_parser.add_argument('--checkpoint', type=str, default=None,
                             help='Path to checkpoint to resume training from')
    
    # Run tests command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--module', type=str, default='all',
                            choices=['all', 'model', 'losses', 'pipeline', 'processing',
                                    'shape_generator', 'token_conversion'],
                            help='Which test module to run')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on a shape')
    inference_parser.add_argument('--input', type=str, required=True,
                                 help='Path to input STL file')
    inference_parser.add_argument('--checkpoint', type=str, required=True,
                                 help='Path to model checkpoint')
    inference_parser.add_argument('--output', type=str, default=None,
                                 help='Path to save output')
    
    return parser

def run_tests(module='all'):
    """Run the specified tests"""
    if module == 'all' or module == 'model':
        from geometrize.test_model import run_all_tests as test_model
        print("Running model tests...")
        test_model()
    
    if module == 'all' or module == 'losses':
        from geometrize.test_losses import run_all_tests as test_losses
        print("Running loss function tests...")
        test_losses()
    
    if module == 'all' or module == 'pipeline':
        from geometrize.test_pipeline import run_all_tests as test_pipeline
        print("Running pipeline tests...")
        test_pipeline()
    
    if module == 'all' or module == 'processing':
        from geometrize.test_processing import test_stl_processing
        print("Running data processing tests...")
        # Just use a sample shape if available
        stl_path = os.path.join(Config.data_dir, "sphere.stl")
        if not os.path.exists(stl_path):
            save_test_shapes()
        test_stl_processing(stl_path)
    
    if module == 'all' or module == 'shape_generator':
        from geometrize.test_shape_generator import run_all_tests as test_shape_generator
        print("Running shape generator tests...")
        test_shape_generator()
    
    if module == 'all' or module == 'token_conversion':
        from geometrize.test_token_conversion import run_all_tests as test_token_conversion
        print("Running token conversion tests...")
        test_token_conversion()

def run_inference(input_file, checkpoint_path, output_path=None):
    """Run inference on an input STL file"""
    from geometrize.data_processing import load_stl_to_pointcloud
    from geometrize.model import PointNetEncoder, TransformerDecoder
    from geometrize.pipeline import reparameterize, convert_tokens_to_operations
    from geometrize.shape_generator import generate_shape_from_tokens
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load the point cloud
    point_cloud = load_stl_to_pointcloud(input_file, Config.num_points)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder = PointNetEncoder(latent_dim=Config.latent_dim).to(device)
    decoder = TransformerDecoder(
        latent_dim=Config.latent_dim,
        token_dim=30,  # From TokenMap (maybe I should parse this from the config?)
        num_bins=Config.num_bins
    ).to(device)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Run inference
    point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode point cloud
        mu, logvar = encoder(point_cloud_tensor)
        
        # Sample latent vector
        z = reparameterize(mu, logvar)
        
        # Decode to tokens
        token_probs, bin_probs, _ = decoder(z)
        
        # Get predicted tokens
        pred_tokens = torch.argmax(token_probs, dim=-1)[0].cpu().numpy()
        pred_bins = torch.argmax(bin_probs, dim=-1)[0].cpu().numpy()
        
    # Convert to operations
    operations = convert_tokens_to_operations(pred_tokens, pred_bins)
    
    # Generate shape from operations
    generated_points = generate_shape_from_tokens(operations)
    
    # Create output directory if specified
    if output_path is None:
        output_path = os.path.join(Config.output_dir, "inference_result.png")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Visualize result
    fig = plt.figure(figsize=(15, 7))
    
    # Original point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c=point_cloud[:, 2],
        cmap='viridis',
        s=5
    )
    ax1.set_title("Original Shape")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_box_aspect([1, 1, 1])
    
    # Generated point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        generated_points[:, 0],
        generated_points[:, 1],
        generated_points[:, 2],
        c=generated_points[:, 2],
        cmap='viridis',
        s=5
    )
    ax2.set_title("Generated Shape")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_box_aspect([1, 1, 1])
    
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")

def main():
    """Main function to parse arguments and run commands"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(Config.data_dir, exist_ok=True)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Handle commands
    if args.command == 'generate':
        print("Generating test shapes...")
        save_test_shapes()
        print("Test shapes generated successfully!")
    
    elif args.command == 'train':
        print("Starting training...")
        # Override config settings with command line arguments
        Config.num_epochs = args.epochs
        Config.batch_size = args.batch_size
        Config.learning_rate = args.learning_rate
        Config.beta = args.beta
        
        # Train the model
        train()
    
    elif args.command == 'test':
        run_tests(args.module)
    
    elif args.command == 'inference':
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            return
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file {args.checkpoint} does not exist.")
            return
        
        run_inference(args.input, args.checkpoint, args.output)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
