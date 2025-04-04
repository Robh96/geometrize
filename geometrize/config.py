class Config:
    # Data settings
    num_points = 2048
    max_seq_length = 50  # Maximum token sequence length for padding
    
    # Model settings
    latent_dim = 256
    num_bins = 20
    
    # Training settings
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    beta = 0.1  # Beta-VAE parameter
    
    # Paths
    data_dir = ".data/"
    checkpoint_dir = ".checkpoints/"
    output_dir = ".outputs/"