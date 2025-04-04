import torch
import torch.nn as nn
import torch.nn.functional as F
from token_map import TokenMap

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # PointNet feature extraction layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # For VAE
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        # x shape: batch_size x n_points x 3
        x = x.transpose(2, 1)  # batch_size x 3 x n_points
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Max pooling (symmetric function for permutation invariance)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        
        # VAE outputs
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim=256, token_dim=30, num_bins=20):
        super().__init__()
        self.token_map = TokenMap()
        self.num_tokens = len(self.token_map.token_to_id)
        self.num_bins = num_bins
        
        # Latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, 512)
        
        # Embedding layers
        # Fix: Change embedding dimensions from 256 to 512 to match positional encoding
        self.token_embedding = nn.Embedding(self.num_tokens, 512)
        self.bin_embedding = nn.Embedding(num_bins, 512)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, 512))  # Max seq len = 100
        
        # Transformer layers
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048)
        self.transformer = nn.TransformerDecoder(
            self.transformer_layer, num_layers=6)
        
        # Output layers
        self.token_predictor = nn.Linear(512, self.num_tokens)
        self.bin_predictor = nn.Linear(512, self.num_bins)
        self.offset_predictor = nn.Linear(512, 1)  # Residual offset
        
    def forward(self, z, target_seq=None, max_len=100):
        batch_size = z.size(0)
        
        # Initial hidden state from latent vector
        hidden = self.latent_to_hidden(z).unsqueeze(0)  # [1, batch_size, 512]
        
        # If target sequence is provided, use teacher forcing regardless of training mode
        if target_seq is not None:
            # Extract target tokens and bins
            target_tokens, target_bins = target_seq[:, :, 0], target_seq[:, :, 1]
            seq_len = target_tokens.size(1)
            
            # Embed tokens and bins
            token_embeds = self.token_embedding(target_tokens)
            bin_embeds = self.bin_embedding(target_bins)
            
            # Combine embeddings
            combined_embeds = token_embeds + bin_embeds
            combined_embeds = combined_embeds + self.positional_encoding[:, :seq_len, :]
            
            # Reshape for transformer: [seq_len, batch_size, embedding_dim]
            combined_embeds = combined_embeds.transpose(0, 1)
            
            # Pass through transformer
            outputs = self.transformer(
                combined_embeds, 
                memory=hidden
            )
            
            # Predict tokens and bins
            token_logits = self.token_predictor(outputs)  # [seq_len, batch, num_tokens]
            bin_logits = self.bin_predictor(outputs)      # [seq_len, batch, num_bins]
            offsets = self.offset_predictor(outputs)      # [seq_len, batch, 1]
            
            # Reshape to [batch, seq_len, num_tokens/bins]
            token_probs = token_logits.transpose(0, 1)
            bin_probs = bin_logits.transpose(0, 1)
            offsets = offsets.transpose(0, 1)
            
            return token_probs, bin_probs, offsets
        
        # If no target sequence provided, use auto-regressive generation
        else:
            # Start with NULL token
            curr_token = torch.full((batch_size, 1), self.token_map.get_id('NULL'), 
                                   device=z.device, dtype=torch.long)
            curr_bin = torch.zeros((batch_size, 1), device=z.device, dtype=torch.long)
            
            token_probs_list = []
            bin_probs_list = []
            offsets_list = []
            
            # Auto-regressive generation
            for i in range(max_len):
                # Embed current tokens and bins
                token_embeds = self.token_embedding(curr_token)  # [batch_size, curr_seq_len, 512]
                bin_embeds = self.bin_embedding(curr_bin)        # [batch_size, curr_seq_len, 512]
                
                # Get current sequence length
                curr_seq_len = curr_token.size(1)
                
                # Combine embeddings with positional encoding - FIX HERE
                # Instead of using complex indexing, directly select the positional encoding
                # for the current sequence length
                pos_encode = self.positional_encoding[:, :curr_seq_len, :]  # [1, curr_seq_len, 512]
                
                # Add the embeddings
                combined_embeds = token_embeds + bin_embeds + pos_encode  # [batch_size, curr_seq_len, 512]
                
                # Reshape for transformer: [seq_len, batch_size, embedding_dim]
                combined_embeds = combined_embeds.transpose(0, 1)  # [curr_seq_len, batch_size, 512]
                
                # Pass through transformer
                output = self.transformer(
                    combined_embeds,
                    memory=hidden
                )
                
                # Get the last output (current step)
                last_output = output[-1]  # [batch_size, 512]
                
                # Predict next token and bin
                token_logit = self.token_predictor(last_output)  # [batch_size, num_tokens]
                bin_logit = self.bin_predictor(last_output)      # [batch_size, num_bins]
                offset = self.offset_predictor(last_output)      # [batch_size, 1]
                
                token_probs_list.append(token_logit.unsqueeze(1))
                bin_probs_list.append(bin_logit.unsqueeze(1))
                offsets_list.append(offset.unsqueeze(1))
                
                # Get next token and bin (greedy decoding)
                next_token = torch.argmax(token_logit, dim=-1, keepdim=True)
                next_bin = torch.argmax(bin_logit, dim=-1, keepdim=True)
                
                # Append to current sequence
                curr_token = torch.cat([curr_token, next_token], dim=1)
                curr_bin = torch.cat([curr_bin, next_bin], dim=1)
                
                # Stop if all sequences in batch predict END token
                if (next_token == self.token_map.get_id('NULL')).all():
                    break
            
            # Combine results
            token_probs = torch.cat(token_probs_list, dim=1)
            bin_probs = torch.cat(bin_probs_list, dim=1)
            offsets = torch.cat(offsets_list, dim=1)
            
            return token_probs, bin_probs, offsets