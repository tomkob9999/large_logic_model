import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MolecularVocabulary:
    """4-bit molecular vocabulary with 16 tokens"""
    
    def __init__(self, encoding_low=0.1, encoding_high=0.9):
        self.chunk_size = 4  # 4 binary variables per chunk
        self.vocab_size = 16  # 2^4 = 16 possible tokens
        self.encoding_low = encoding_low
        self.encoding_high = encoding_high
        
        # Create all 16 possible 4-bit combinations
        self.molecules = []
        for i in range(16):
            binary = format(i, '04b')
            molecule = [int(bit) for bit in binary]
            self.molecules.append(molecule)
    
    def encode_sequence(self, binary_array):
        """Convert binary array to sequence of molecular tokens"""
        # Pad to multiple of 4
        padded_length = ((len(binary_array) + 3) // 4) * 4
        padded = np.pad(binary_array, (0, padded_length - len(binary_array)), 'constant')
        
        # Split into 4-bit chunks
        chunks = padded.reshape(-1, 4)
        
        # Convert to token indices
        tokens = []
        for chunk in chunks:
            token_id = sum(chunk[i] * (2**(3-i)) for i in range(4))
            tokens.append(token_id)
        
        return tokens
    
    def encode_continuous(self, binary_array):
        """Encode binary values using 0.1/0.9 strategy"""
        return np.where(binary_array == 0, self.encoding_low, self.encoding_high)

class ProteinInspiredDNFGenerator(nn.Module):
    """DNF Generator with configurable embedding vector dimensions"""
    
    def __init__(self, num_variables, embedding_dim=4, d_model=64, nhead=4, num_layers=2):
        super(ProteinInspiredDNFGenerator, self).__init__()
        
        self.vocab = MolecularVocabulary()
        self.num_variables = num_variables
        self.embedding_dim = embedding_dim  # Vector dimensionality (4, 6, or 8)
        self.d_model = d_model
        self.seq_length = (num_variables + 3) // 4
        
        # Molecular token embedding: 16 tokens → embedding_dim dimensions
        self.token_embedding = nn.Embedding(self.vocab.vocab_size, embedding_dim)
        
        # Project to transformer dimension if needed
        if embedding_dim != d_model:
            self.projection = nn.Linear(embedding_dim, d_model)
        else:
            self.projection = nn.Identity()
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_pos_encoding(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, 1)
    
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, binary_inputs):
        batch_size = binary_inputs.size(0)
        
        # Convert to molecular sequences
        molecular_sequences = []
        for i in range(batch_size):
            binary_array = binary_inputs[i].cpu().numpy().astype(int)
            tokens = self.vocab.encode_sequence(binary_array)
            molecular_sequences.append(tokens)
        
        # Pad sequences
        max_len = max(len(seq) for seq in molecular_sequences)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, seq in enumerate(molecular_sequences):
            padded[i, :len(seq)] = torch.tensor(seq)
        
        padded = padded.to(binary_inputs.device)
        
        # Token embedding: [batch, seq_len] → [batch, seq_len, embedding_dim]
        embedded = self.token_embedding(padded)
        
        # Project to transformer dimension: [batch, seq_len, embedding_dim] → [batch, seq_len, d_model]
        projected = self.projection(embedded)
        
        # Add positional encoding
        seq_len = projected.size(1)
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0)
        projected = projected + pos_enc
        
        # Transformer processing
        encoded = self.transformer(projected)
        
        # Global average pooling and classification
        pooled = encoded.mean(dim=1)  # [batch, d_model]
        logits = self.classifier(pooled)  # [batch, 1]
        
        return logits

class DNFPatternGenerator:
    """Generate ground truth DNF patterns for evaluation"""
    
    def __init__(self, num_variables=40):
        self.num_variables = num_variables
        self.patterns = [
            [0],           # pos1
            [2, 3],        # pos3 AND pos4
            [5, 6, 7],     # pos6 AND pos7 AND pos8
            [10, 11],      # pos11 AND pos12
            [15, 20, 25],  # pos16 AND pos21 AND pos26
        ]
    
    def generate_dataset(self, n_samples=1000):
        samples = []
        labels = []
        
        # Generate positive samples
        for _ in range(n_samples // 2):
            # Choose random pattern
            pattern = np.random.choice(len(self.patterns))
            sample = np.zeros(self.num_variables, dtype=int)
            
            # Set pattern positions to 1
            for pos in self.patterns[pattern]:
                sample[pos] = 1
            
            # Add random noise
            noise_positions = np.random.choice(self.num_variables, 
                                             size=np.random.randint(0, 5), replace=False)
            for pos in noise_positions:
                if pos not in self.patterns[pattern]:
                    sample[pos] = np.random.randint(0, 2)
            
            samples.append(sample)
            labels.append(1)
        
        # Generate negative samples
        for _ in range(n_samples // 2):
            sample = np.random.randint(0, 2, self.num_variables)
            
            # Ensure it doesn't match any pattern
            matches_pattern = False
            for pattern in self.patterns:
                if all(sample[pos] == 1 for pos in pattern):
                    matches_pattern = True
                    break
            
            if not matches_pattern:
                samples.append(sample)
                labels.append(0)
        
        return np.array(samples), np.array(labels)

def train_model(model, train_data, train_labels, epochs=20, lr=0.001):
    """Train the DNF model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    train_tensor = torch.FloatTensor(train_data)
    labels_tensor = torch.FloatTensor(train_labels).unsqueeze(1)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(train_tensor)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

def evaluate_model(model, test_data, test_labels):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_data)
        logits = model(test_tensor)
        predictions = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary', zero_division=0
    )
    
    # Calculate false positive and false negative rates
    tn = np.sum((test_labels == 0) & (predictions == 0))
    fp = np.sum((test_labels == 0) & (predictions == 1))
    fn = np.sum((test_labels == 1) & (predictions == 0))
    tp = np.sum((test_labels == 1) & (predictions == 1))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }

def main():
    """Main comparison function"""
    print("🧬 Protein-Inspired DNF Generator Comparison")
    print("=" * 60)

    n_trains = 10000
    n_tests = 2000
    n_variables = 400
    
    # Generate dataset
    print("📊 Generating Dataset...")
    pattern_gen = DNFPatternGenerator(num_variables=n_variables)
    train_data, train_labels = pattern_gen.generate_dataset(n_samples=n_trains)
    test_data, test_labels = pattern_gen.generate_dataset(n_samples=n_tests)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Positive samples: {np.sum(train_labels)}/{len(train_labels)}")
    
    # Test both embedding dimensions
    embedding_dims = [4, 6]
    results = {}
    
    for dim in embedding_dims:
        print(f"\n🔬 Training {dim}D Embedding Model...")
        
        # Create and train model
        model = ProteinInspiredDNFGenerator(
            # num_variables=40, 
            num_variables=n_variables, 
            embedding_dim=dim,
            d_model=64,
            nhead=4,
            num_layers=2
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Train model
        start_time = time.time()
        model = train_model(model, train_data, train_labels, epochs=20)
        training_time = time.time() - start_time
        
        # Evaluate model
        metrics = evaluate_model(model, test_data, test_labels)
        metrics['training_time'] = training_time
        metrics['parameters'] = sum(p.numel() for p in model.parameters())
        
        results[f"{dim}D"] = metrics
        
        print(f"Results for {dim}D embedding:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.3f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.3f}")
        print(f"  Training Time: {metrics['training_time']:.2f}s")
        print(f"  Parameters: {metrics['parameters']}")
    
    # Final comparison
    print("\n" + "=" * 60)
    print("📈 FINAL COMPARISON")
    print("=" * 60)
    
    print("\nPerformance Comparison:")
    print(f"{'Metric':<20} {'4D':<10} {'6D':<10} {'Better':<10}")
    print("-" * 50)
    
    metrics_to_compare = ['accuracy', 'f1', 'precision', 'recall']
    for metric in metrics_to_compare:
        val_4d = results['4D'][metric]
        val_6d = results['6D'][metric]
        better = '4D' if val_4d > val_6d else '6D' if val_6d > val_4d else 'Tie'
        print(f"{metric:<20} {val_4d:<10.3f} {val_6d:<10.3f} {better:<10}")
    
    print(f"{'training_time':<20} {results['4D']['training_time']:<10.2f} {results['6D']['training_time']:<10.2f}")
    print(f"{'parameters':<20} {results['4D']['parameters']:<10} {results['6D']['parameters']:<10}")
    
    # Recommendation
    print("\n🎯 RECOMMENDATION:")
    if results['4D']['f1'] >= results['6D']['f1'] * 0.95:  # Within 5%
        print("✅ 4D Embedding is recommended")
        print("  Reasons: Similar performance with fewer parameters and faster training")
    else:
        print("✅ 6D Embedding is recommended") 
        print("  Reasons: Superior performance justifies additional complexity")
    
    print("\n🧬 Default Configuration:")
    print("  Encoding: 0.1/0.9")
    print("  Chunk size: 4-bit (16 tokens)")
    print(f"  Embedding dimension: {'4D' if results['4D']['f1'] >= results['6D']['f1'] * 0.95 else '6D'}")
    print("  Each token: 64-bit float elements")

if __name__ == "__main__":
    main()
