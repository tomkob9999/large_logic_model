import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sklearn.metrics as metrics
import wandb

class DNFTrainingPipeline:
    def __init__(self, model, config):
        """
        Initialize training pipeline with advanced metrics tracking
        
        Args:
            model (nn.Module): DNF Transformer Encoder model
            config (dict): Configuration for training
        """
        self.model = model
        self.config = config
        
        # Optimizers with adaptive learning rate
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.binary_loss = nn.BCEWithLogitsLoss()
        
    def custom_loss(self, outputs, targets):
        """
        Custom loss function that penalizes false positives and false negatives
        
        Args:
            outputs (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth
        
        Returns:
            torch.Tensor: Composite loss
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(outputs, targets)
        
        # Binary classification loss
        binary_loss = self.binary_loss(outputs, targets)
        
        # False Positive Penalty
        fp_mask = (outputs > 0.5) & (targets < 0.5)
        fp_penalty = torch.mean(outputs[fp_mask])
        
        # False Negative Penalty
        fn_mask = (outputs < 0.5) & (targets > 0.5)
        fn_penalty = torch.mean(1 - outputs[fn_mask])
        
        # Combine losses
        total_loss = (
            recon_loss + 
            binary_loss + 
            self.config.get('fp_weight', 1.0) * fp_penalty + 
            self.config.get('fn_weight', 1.0) * fn_penalty
        )
        
        return total_loss
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch with detailed metrics tracking
        
        Args:
            dataloader (DataLoader): Training data loader
        
        Returns:
            dict: Epoch training metrics
        """
        self.model.train()
        epoch_losses = []
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            inputs = batch['input']
            targets = batch['target']
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.custom_loss(outputs, targets)
            epoch_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Collect predictions for metrics
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Compute epoch metrics
        metrics_dict = self.compute_metrics(
            np.array(all_preds), 
            np.array(all_targets)
        )
        metrics_dict['avg_loss'] = np.mean(epoch_losses)
        
        return metrics_dict
    
    def validate(self, dataloader):
        """
        Validate model performance
        
        Args:
            dataloader (DataLoader): Validation data loader
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input']
                targets = batch['target']
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.custom_loss(outputs, targets)
                val_losses.append(loss.item())
                
                # Collect predictions
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute validation metrics
        metrics_dict = self.compute_metrics(
            np.array(all_preds), 
            np.array(all_targets)
        )
        metrics_dict['avg_loss'] = np.mean(val_losses)
        
        return metrics_dict
    
    def compute_metrics(self, preds, targets):
        """
        Compute comprehensive performance metrics
        
        Args:
            preds (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth
        
        Returns:
            dict: Performance metrics
        """
        # Compute confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(targets, preds).ravel()
        
        # Compute metrics
        metrics_dict = {
            'accuracy': metrics.accuracy_score(targets, preds),
            'precision': metrics.precision_score(targets, preds),
            'recall': metrics.recall_score(targets, preds),
            'f1_score': metrics.f1_score(targets, preds),
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp),
            'true_positive_rate': tp / (tp + fn),
            'true_negative_rate': tn / (tn + fp)
        }
        
        return metrics_dict
    
    def train(self, train_dataloader, val_dataloader, epochs=50):
        """
        Full training process with logging and early stopping
        
        Args:
            train_dataloader (DataLoader): Training data loader
            val_dataloader (DataLoader): Validation data loader
            epochs (int): Number of training epochs
        
        Returns:
            dict: Final training results
        """
        # Initialize wandb for experiment tracking
        wandb.init(
            project="DNF-Generator",
            config=self.config
        )
        
        best_f1 = 0
        early_stop_counter = 0
        
        for epoch in range(epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validate
            val_metrics = self.validate(val_dataloader)
            
            # Log metrics
            wandb.log({
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['avg_loss'])
            
            # Early stopping with F1 score
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                early_stop_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                early_stop_counter += 1
            
            # Stop if no improvement
            if early_stop_counter >= 10:
                print("Early stopping triggered")
                break
        
        # Finish wandb run
        wandb.finish()
        
        return {
            'best_f1_score': best_f1,
            'final_metrics': val_metrics
        }

# Example usage
if __name__ == "__main__":
    # Assume we have our DNFTransformerEncoder from previous implementation
    from models.dnf_transformer import DNFTransformerEncoder
    
    # Configuration
    config = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'fp_weight': 1.0,  # False Positive penalty weight
        'fn_weight': 1.0,  # False Negative penalty weight
        'num_variables': 10000,
        'embedding_dim': 8
    }
    
    # Initialize model
    model = DNFTransformerEncoder(
        num_variables=config['num_variables'], 
        embedding_dim=config['embedding_dim']
    )
    
    # Initialize training pipeline
    trainer = DNFTrainingPipeline(model, config)
    
    # Placeholder for actual data loaders
    train_dataloader = None  # Replace with actual DataLoader
    val_dataloader = None   # Replace with actual DataLoader
    
    # Train
    results = trainer.train(train_dataloader, val_dataloader)