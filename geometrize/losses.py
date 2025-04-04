import torch
import torch.nn.functional as F

def vae_loss(recon_token_probs, recon_bin_probs, target_tokens, target_bins, mu, logvar):
    """Combined VAE loss with reconstruction and KL divergence components."""
    # Token prediction loss - FIX: use reshape instead of view
    token_loss = F.cross_entropy(
        recon_token_probs.reshape(-1, recon_token_probs.size(-1)), 
        target_tokens.reshape(-1)
    )
    
    # Bin prediction loss - FIX: use reshape instead of view
    bin_mask = target_tokens != 29  # Non-NULL tokens
    bin_loss = F.cross_entropy(
        recon_bin_probs.reshape(-1, recon_bin_probs.size(-1))[bin_mask.reshape(-1)],
        target_bins.reshape(-1)[bin_mask.reshape(-1)]
    )
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta weight for KL term (from Beta-VAE)
    beta = 0.1
    
    return token_loss + bin_loss + beta * kl_div

def hausdorff_distance(points1, points2):
    """Calculate Hausdorff distance between two point clouds.
    
    Args:
        points1: Tensor of shape [B, N, 3] - first point cloud
        points2: Tensor of shape [B, M, 3] - second point cloud
        
    Returns:
        Tensor of shape [B] - Hausdorff distance for each batch item
    """
    batch_size = points1.shape[0]
    n_points1 = points1.shape[1]
    n_points2 = points2.shape[1]
    
    # Reshape for broadcasting
    p1 = points1.unsqueeze(2)  # [B, N, 1, 3]
    p2 = points2.unsqueeze(1)  # [B, 1, M, 3]
    
    # Calculate pairwise distances between all points
    # Output shape: [B, N, M]
    distances = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=3) + 1e-8)
    
    # Forward Hausdorff: for each point in p1, find min distance to p2
    # Then take max of those min distances
    forward_hausdorff = torch.max(torch.min(distances, dim=2)[0], dim=1)[0]
    
    # Backward Hausdorff: for each point in p2, find min distance to p1
    # Then take max of those min distances
    backward_hausdorff = torch.max(torch.min(distances, dim=1)[0], dim=1)[0]
    
    # Symmetric Hausdorff distance is the max of forward and backward
    hausdorff = torch.max(forward_hausdorff, backward_hausdorff)
    
    return hausdorff