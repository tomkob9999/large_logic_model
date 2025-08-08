import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class DNFTransformerEncoder(nn.Module):
    def __init__(self, 
                 num_variables=10000, 
                 embedding_dim=8, 
                 num_heads=8, 
                 num_layers=6, 
                 dropout=0.1):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(2, embedding_dim, dtype=torch.float64)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Autoencoder compression layers
        self.compression_layer = nn.Sequential(
            nn.Linear(num_variables * embedding_dim, num_variables // 10),
            nn.ReLU(),
            nn.Linear(num_variables // 10, num_variables * embedding_dim)
        )
        
        # DNF clause reduction layer
        self.dnf_reduction_layer = nn.Sequential(
            nn.Linear(num_variables * embedding_dim, num_variables // 10),
            nn.ReLU(),
            nn.Linear(num_variables // 10, num_variables)
        )
    
    def forward(self, x):
        # x shape: [batch_size, num_variables]
        
        # Convert to binary indices for embedding
        binary_indices = (x > 0.5).long()
        
        # Embed the binary indices
        embedded = self.embedding(binary_indices)
        
        # Add positional encoding
        embedded = self.positional_encoding(embedded)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(embedded)
        
        # Flatten the encoded representation
        flattened = encoded.view(encoded.size(0), -1)
        
        # Autoencoder compression
        compressed = self.compression_layer(flattened)
        
        # DNF clause reduction
        dnf_reduced = self.dnf_reduction_layer(compressed)
        
        return dnf_reduced

class DNFLoss(nn.Module):
    def __init__(self, reduction_factor=0.1):
        super().__init__()
        self.reduction_factor = reduction_factor
    
    def forward(self, original, reconstructed):
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(original, reconstructed)
        
        # Sparsity loss to encourage compact representation
        sparsity_loss = torch.mean(torch.abs(reconstructed))
        
        # Redundancy penalty
        redundancy_penalty = self.calculate_redundancy_penalty(reconstructed)
        
        # Combined loss
        total_loss = (reconstruction_loss + 
                      self.reduction_factor * sparsity_loss + 
                      redundancy_penalty)
        
        return total_loss
    
    def calculate_redundancy_penalty(self, tensor):
        # Calculate correlation between different clauses
        correlation_matrix = torch.corrcoef(tensor.T)
        
        # Penalize high correlation between clauses
        redundancy_penalty = torch.sum(torch.abs(correlation_matrix - torch.eye(correlation_matrix.size(0))))
        
        return redundancy_penalty

# Example usage
if __name__ == "__main__":
    # Simulate a large binary dataset
    batch_size = 64
    num_variables = 10000
    
    # Generate random binary input
    input_data = torch.randint(0, 2, (batch_size, num_variables), dtype=torch.float64)
    
    # Initialize the model
    model = DNFTransformerEncoder(num_variables=num_variables)
    
    # Loss function
    criterion = DNFLoss()
    
    # Forward pass
    output = model(input_data)
    
    # Calculate loss
    loss = criterion(input_data, output)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item()}")