import torch
import numpy as np
import json
import logging

class DNFTestRunner:
    def __init__(self, config):
        """
        Initialize DNF Test Runner with comprehensive logging
        
        Args:
            config (dict): Configuration parameters
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.num_variables = config.get('num_variables', 10000)
        self.embedding_dim = config.get('embedding_dim', 8)
        
        # Seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def generate_test_dataset(self, num_samples=100000, complexity=0.3):
        """
        Generate a more structured synthetic dataset with controlled complexity
        
        Args:
            num_samples (int): Number of samples to generate
            complexity (float): Complexity of logical relationships
        
        Returns:
            torch.Tensor: Test dataset
        """
        # Generate base random binary data
        data = torch.randint(0, 2, (num_samples, self.num_variables), dtype=torch.float)
        
        # Introduce some structured logical relationships
        for i in range(0, self.num_variables, 10):
            # Create groups of correlated variables
            group_size = min(5, self.num_variables - i)
            group = data[:, i:i+group_size]
            
            # Introduce some logical dependencies
            if group_size > 1:
                # Make subsequent variables dependent on previous ones
                for j in range(1, group_size):
                    group[:, j] = torch.logical_and(
                        group[:, j-1], 
                        torch.randint(0, 2, (num_samples,), dtype=torch.float)
                    ).float()
        
        return data
    
    def compute_metrics(self, ground_truth, predictions):
        """
        Compute comprehensive performance metrics
        
        Args:
            ground_truth (torch.Tensor): Original input data
            predictions (torch.Tensor): Model predictions
        
        Returns:
            dict: Detailed performance metrics
        """
        # Flatten tensors for metric computation
        gt_flat = ground_truth.view(-1)
        pred_flat = predictions.view(-1)
        
        # Compute confusion matrix components
        true_positives = torch.sum((gt_flat == 1) & (pred_flat == 1)).item()
        true_negatives = torch.sum((gt_flat == 0) & (pred_flat == 0)).item()
        false_positives = torch.sum((gt_flat == 0) & (pred_flat == 1)).item()
        false_negatives = torch.sum((gt_flat == 1) & (pred_flat == 0)).item()
        
        # Compute metrics
        total_samples = len(gt_flat)
        
        metrics = {
            # Basic Counts
            'total_samples': total_samples,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            
            # Rates
            'true_positive_rate': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
            'true_negative_rate': true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0,
            'false_positive_rate': false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0,
            'false_negative_rate': false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0,
            
            # Precision and Recall
            'precision': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
            'recall': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
        }
        
        # Compute F1 Score
        metrics['f1_score'] = (
            2 * metrics['precision'] * metrics['recall'] / 
            (metrics['precision'] + metrics['recall'])
        ) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        return metrics
    
    def run_test(self):
        """
        Run comprehensive test with detailed metrics tracking
        
        Returns:
            dict: Comprehensive test results
        """
        # Generate test dataset
        test_data = self.generate_test_dataset()
        
        # Simulate model prediction (binary classification)
        predictions = torch.randint(0, 2, test_data.shape, dtype=torch.float)
        
        # Compute metrics
        metrics = self.compute_metrics(test_data, predictions)
        
        # Print results
        print("\n===== DNF Generator Test Results =====")
        print(f"Total Samples: {metrics['total_samples']}")
        
        print("\n--- Confusion Matrix Components ---")
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"True Negatives:  {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        
        print("\n--- Performance Metrics ---")
        print(f"F1 Score:              {metrics['f1_score']:.4f}")
        print(f"Precision:             {metrics['precision']:.4f}")
        print(f"Recall:                {metrics['recall']:.4f}")
        
        print("\n--- Rates ---")
        print(f"True Positive Rate:    {metrics['true_positive_rate']:.4f}")
        print(f"True Negative Rate:    {metrics['true_negative_rate']:.4f}")
        print(f"False Positive Rate:   {metrics['false_positive_rate']:.4f}")
        print(f"False Negative Rate:   {metrics['false_negative_rate']:.4f}")
        
        # Save results to JSON
        with open('test_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

# Main execution
if __name__ == "__main__":
    # Configuration
    config = {
        'num_variables': 10000,
        'embedding_dim': 8
    }
    
    # Run comprehensive test
    test_runner = DNFTestRunner(config)
    test_results = test_runner.run_test()