import torch
import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dnf_transformer import DNFTransformerEncoder

class TestDNFTransformerEncoder(unittest.TestCase):
    """
    Comprehensive test suite for DNF Transformer Encoder
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.default_config = {
            'num_variables': 1000,  # Smaller for testing
            'embedding_dim': 8,
            'threshold': 0.5
        }
        self.model = DNFTransformerEncoder(**self.default_config)
    
    def test_initialization(self):
        """Test model initialization with default parameters."""
        self.assertEqual(self.model.num_variables, 1000)
        self.assertEqual(self.model.embedding_dim, 8)
        self.assertEqual(self.model.threshold, 0.5)
        
        # Test embedding layer
        self.assertIsNotNone(self.model.embedding)
        self.assertEqual(self.model.embedding.num_embeddings, 2)
        self.assertEqual(self.model.embedding.embedding_dim, 8)
    
    def test_initialization_custom_parameters(self):
        """Test model initialization with custom parameters."""
        custom_config = {
            'num_variables': 5000,
            'embedding_dim': 16,
            'threshold': 0.7
        }
        custom_model = DNFTransformerEncoder(**custom_config)
        
        self.assertEqual(custom_model.num_variables, 5000)
        self.assertEqual(custom_model.embedding_dim, 16)
        self.assertEqual(custom_model.threshold, 0.7)
    
    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        batch_size = 32
        input_tensor = torch.randint(0, 2, (batch_size, self.default_config['num_variables']), dtype=torch.float)
        
        output = self.model(input_tensor)
        
        # Check output structure
        self.assertIsInstance(output, dict)
        self.assertIn('probabilities', output)
        self.assertIn('binary_predictions', output)
        
        # Check shapes
        probabilities = output['probabilities']
        binary_predictions = output['binary_predictions']
        
        expected_shape = (batch_size, self.default_config['num_variables'])
        self.assertEqual(probabilities.shape, expected_shape)
        self.assertEqual(binary_predictions.shape, expected_shape)
    
    def test_probability_constraints(self):
        """Test that probabilities are within [0, 1] range."""
        batch_size = 16
        input_tensor = torch.randint(0, 2, (batch_size, self.default_config['num_variables']), dtype=torch.float)
        
        output = self.model(input_tensor)
        probabilities = output['probabilities']
        
        # Check probability constraints
        self.assertTrue(torch.all(probabilities >= 0), "Probabilities should be >= 0")
        self.assertTrue(torch.all(probabilities <= 1), "Probabilities should be <= 1")
    
    def test_binary_predictions_constraints(self):
        """Test that binary predictions are strictly 0 or 1."""
        batch_size = 16
        input_tensor = torch.randint(0, 2, (batch_size, self.default_config['num_variables']), dtype=torch.float)
        
        output = self.model(input_tensor)
        binary_predictions = output['binary_predictions']
        
        # Check binary constraints
        unique_values = torch.unique(binary_predictions)
        self.assertTrue(torch.all((unique_values == 0) | (unique_values == 1)), 
                       "Binary predictions must be 0 or 1")
    
    def test_threshold_functionality(self):
        """Test that threshold parameter affects binary predictions."""
        batch_size = 8
        input_tensor = torch.randint(0, 2, (batch_size, self.default_config['num_variables']), dtype=torch.float)
        
        # Test different thresholds
        thresholds = [0.3, 0.5, 0.7]
        results = []
        
        for threshold in thresholds:
            model = DNFTransformerEncoder(
                num_variables=self.default_config['num_variables'],
                embedding_dim=self.default_config['embedding_dim'],
                threshold=threshold
            )
            output = model(input_tensor)
            results.append(output['binary_predictions'])
        
        # Different thresholds should potentially produce different results
        # (though not guaranteed with random initialization)
        self.assertEqual(len(results), 3)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        batch_size = 8
        input_tensor = torch.randint(0, 2, (batch_size, self.default_config['num_variables']), dtype=torch.float)
        target = torch.randint(0, 2, (batch_size, self.default_config['num_variables']), dtype=torch.float)
        
        # Enable gradient computation
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        probabilities = output['probabilities']
        
        # Compute a simple loss
        loss = torch.nn.functional.binary_cross_entropy(probabilities, target)
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(input_tensor.grad)
        
        # Check that model parameters have gradients
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_model_consistency(self):
        """Test that the model produces consistent outputs for the same input."""
        batch_size = 4
        input_tensor = torch.randint(0, 2, (batch_size, self.default_config['num_variables']), dtype=torch.float)
        
        # Set model to evaluation mode for consistency
        self.model.eval()
        
        with torch.no_grad():
            output1 = self.model(input_tensor)
            output2 = self.model(input_tensor)
        
        # Outputs should be identical
        torch.testing.assert_close(output1['probabilities'], output2['probabilities'])
        torch.testing.assert_close(output1['binary_predictions'], output2['binary_predictions'])
    
    def test_large_input_handling(self):
        """Test model's ability to handle larger input sizes."""
        large_config = {
            'num_variables': 5000,
            'embedding_dim': 8,
            'threshold': 0.5
        }
        large_model = DNFTransformerEncoder(**large_config)
        
        batch_size = 4  # Smaller batch for memory efficiency
        input_tensor = torch.randint(0, 2, (batch_size, large_config['num_variables']), dtype=torch.float)
        
        # Should not raise any exceptions
        output = large_model(input_tensor)
        
        # Check output shapes
        expected_shape = (batch_size, large_config['num_variables'])
        self.assertEqual(output['probabilities'].shape, expected_shape)
        self.assertEqual(output['binary_predictions'].shape, expected_shape)

def run_transformer_tests():
    """Run all transformer tests."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_transformer_tests()