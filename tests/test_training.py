import torch
import torch.nn as nn
import unittest
import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import DNFTrainingPipeline
from dnf_transformer import DNFTransformerEncoder

class MockDataLoader:
    """Mock DataLoader for testing purposes."""
    
    def __init__(self, num_batches=5, batch_size=16, num_variables=1000):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_variables = num_variables
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        batch = {
            'input': torch.randint(0, 2, (self.batch_size, self.num_variables), dtype=torch.float),
            'target': torch.randint(0, 2, (self.batch_size, self.num_variables), dtype=torch.float)
        }
        self.current_batch += 1
        return batch

class TestDNFTrainingPipeline(unittest.TestCase):
    """
    Comprehensive test suite for DNF Training Pipeline
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model_config = {
            'num_variables': 1000,
            'embedding_dim': 8,
            'threshold': 0.5
        }
        
        self.training_config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'fp_weight': 1.0,
            'fn_weight': 1.0
        }
        
        self.model = DNFTransformerEncoder(**self.model_config)
        self.trainer = DNFTrainingPipeline(self.model, self.training_config)
    
    def test_initialization(self):
        """Test training pipeline initialization."""
        # Check model assignment
        self.assertIsNotNone(self.trainer.model)
        self.assertEqual(self.trainer.model, self.model)
        
        # Check optimizer initialization
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.AdamW)
        
        # Check scheduler initialization
        self.assertIsNotNone(self.trainer.scheduler)
        self.assertIsInstance(self.trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        # Check loss function
        self.assertIsNotNone(self.trainer.bce_loss)
        self.assertIsInstance(self.trainer.bce_loss, nn.BCELoss)
        
        # Check configuration
        self.assertEqual(self.trainer.config, self.training_config)
    
    def test_custom_loss_computation(self):
        """Test custom loss function computation."""
        batch_size = 16
        num_variables = self.model_config['num_variables']
        
        # Create mock outputs and targets
        mock_outputs = {
            'probabilities': torch.rand(batch_size, num_variables),
            'binary_predictions': torch.randint(0, 2, (batch_size, num_variables), dtype=torch.float)
        }
        mock_targets = torch.randint(0, 2, (batch_size, num_variables), dtype=torch.float)
        
        # Compute loss
        loss = self.trainer.custom_loss(mock_outputs, mock_targets)
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertTrue(loss.item() >= 0)  # Non-negative loss
    
    def test_metrics_computation(self):
        """Test performance metrics computation."""
        # Create mock predictions and targets
        batch_size = 100
        preds = np.random.randint(0, 2, batch_size)
        targets = np.random.randint(0, 2, batch_size)
        
        # Compute metrics
        metrics = self.trainer.compute_metrics(preds, targets)
        
        # Check metric keys
        expected_keys = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'false_positive_rate', 'false_negative_rate',
            'true_positive_rate', 'true_negative_rate'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (float, np.floating))
            self.assertTrue(0 <= metrics[key] <= 1)  # Metrics should be in [0, 1]
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        # Create mock dataloader
        mock_dataloader = MockDataLoader(num_batches=3, batch_size=8, 
                                       num_variables=self.model_config['num_variables'])
        
        # Train one epoch
        metrics = self.trainer.train_epoch(mock_dataloader)
        
        # Check returned metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('avg_loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check metric values
        self.assertTrue(metrics['avg_loss'] >= 0)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['f1_score'] <= 1)
    
    def test_validate(self):
        """Test validation functionality."""
        # Create mock dataloader
        mock_dataloader = MockDataLoader(num_batches=2, batch_size=8,
                                       num_variables=self.model_config['num_variables'])
        
        # Validate
        metrics = self.trainer.validate(mock_dataloader)
        
        # Check returned metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('avg_loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check metric values
        self.assertTrue(metrics['avg_loss'] >= 0)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['f1_score'] <= 1)
    
    def test_optimizer_step(self):
        """Test that optimizer updates model parameters."""
        # Get initial parameter values
        initial_params = [param.clone() for param in self.model.parameters()]
        
        # Create mock data
        batch_size = 8
        num_variables = self.model_config['num_variables']
        
        mock_outputs = {
            'probabilities': torch.rand(batch_size, num_variables, requires_grad=True),
            'binary_predictions': torch.randint(0, 2, (batch_size, num_variables), dtype=torch.float)
        }
        mock_targets = torch.randint(0, 2, (batch_size, num_variables), dtype=torch.float)
        
        # Compute loss and perform backward pass
        loss = self.trainer.custom_loss(mock_outputs, mock_targets)
        self.trainer.optimizer.zero_grad()
        loss.backward()
        self.trainer.optimizer.step()
        
        # Check that parameters have changed
        current_params = list(self.model.parameters())
        
        # At least some parameters should have changed
        params_changed = False
        for initial, current in zip(initial_params, current_params):
            if not torch.equal(initial, current):
                params_changed = True
                break
        
        self.assertTrue(params_changed, "Model parameters should change after optimizer step")
    
    def test_loss_components(self):
        """Test individual components of the custom loss function."""
        batch_size = 16
        num_variables = self.model_config['num_variables']
        
        # Create specific test cases
        # Case 1: Perfect predictions
        perfect_probs = torch.ones(batch_size, num_variables) * 0.9
        perfect_preds = torch.ones(batch_size, num_variables)
        perfect_targets = torch.ones(batch_size, num_variables)
        
        perfect_outputs = {
            'probabilities': perfect_probs,
            'binary_predictions': perfect_preds
        }
        
        perfect_loss = self.trainer.custom_loss(perfect_outputs, perfect_targets)
        
        # Case 2: Random predictions
        random_probs = torch.rand(batch_size, num_variables)
        random_preds = torch.randint(0, 2, (batch_size, num_variables), dtype=torch.float)
        random_targets = torch.randint(0, 2, (batch_size, num_variables), dtype=torch.float)
        
        random_outputs = {
            'probabilities': random_probs,
            'binary_predictions': random_preds
        }
        
        random_loss = self.trainer.custom_loss(random_outputs, random_targets)
        
        # Perfect predictions should have lower loss (generally)
        self.assertIsInstance(perfect_loss, torch.Tensor)
        self.assertIsInstance(random_loss, torch.Tensor)
        self.assertTrue(perfect_loss.item() >= 0)
        self.assertTrue(random_loss.item() >= 0)
    
    def test_configuration_parameters(self):
        """Test that configuration parameters are properly used."""
        # Test with different configuration
        custom_config = {
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'fp_weight': 2.0,
            'fn_weight': 1.5
        }
        
        custom_trainer = DNFTrainingPipeline(self.model, custom_config)
        
        # Check that configuration is stored
        self.assertEqual(custom_trainer.config, custom_config)
        
        # Check optimizer parameters
        optimizer_lr = custom_trainer.optimizer.param_groups[0]['lr']
        optimizer_weight_decay = custom_trainer.optimizer.param_groups[0]['weight_decay']
        
        self.assertEqual(optimizer_lr, custom_config['learning_rate'])
        self.assertEqual(optimizer_weight_decay, custom_config['weight_decay'])

def run_training_tests():
    """Run all training pipeline tests."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_training_tests()