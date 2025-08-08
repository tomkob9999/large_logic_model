import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from src.models.dnf_transformer import DNFTransformerEncoder
from src.train import DNFTrainingPipeline
from src.dnf_clause_reducer import DNFClauseReducer
from src.data.dataset import DNFDataset

class DNFGeneratorIntegrator:
    def __init__(self, config):
        """
        Integrate all components of the DNF Generator
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.num_variables = config.get('num_variables', 10000)
        self.embedding_dim = config.get('embedding_dim', 8)
        
        # Initialize core components
        self.model = self._initialize_model()
        self.training_pipeline = self._initialize_training_pipeline()
        self.clause_reducer = self._initialize_clause_reducer()
    
    def _initialize_model(self):
        """
        Initialize Transformer Encoder Model
        
        Returns:
            DNFTransformerEncoder: Configured model
        """
        return DNFTransformerEncoder(
            num_variables=self.num_variables,
            embedding_dim=self.embedding_dim
        )
    
    def _initialize_training_pipeline(self):
        """
        Initialize Training Pipeline
        
        Returns:
            DNFTrainingPipeline: Configured training pipeline
        """
        return DNFTrainingPipeline(
            model=self.model, 
            config=self.config
        )
    
    def _initialize_clause_reducer(self):
        """
        Initialize DNF Clause Reducer
        
        Returns:
            DNFClauseReducer: Configured clause reducer
        """
        return DNFClauseReducer(
            num_variables=self.num_variables
        )
    
    def generate_synthetic_dataset(self, num_samples=100000):
        """
        Generate synthetic dataset for testing
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            tuple: Training and validation datasets
        """
        # Generate random binary data
        data = torch.randint(0, 2, (num_samples, self.num_variables), dtype=torch.float)
        
        # Split into training and validation
        train_size = int(0.8 * num_samples)
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        # Create datasets
        train_dataset = DNFDataset(train_data)
        val_dataset = DNFDataset(val_data)
        
        return train_dataset, val_dataset
    
    def train_and_evaluate(self, train_dataset, val_dataset):
        """
        Train the model and evaluate performance
        
        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
        
        Returns:
            dict: Comprehensive performance metrics
        """
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=64, 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=64, 
            shuffle=False
        )
        
        # Train the model
        training_results = self.training_pipeline.train(
            train_dataloader=train_loader, 
            val_dataloader=val_loader
        )
        
        # Extract clauses from trained model
        model_output = self.model(val_dataset.data)
        reduced_clauses = self.clause_reducer.extract_clauses(model_output)
        
        # Evaluate clause coverage
        coverage_metrics = self.clause_reducer.evaluate_dnf_coverage(
            val_dataset.data, 
            reduced_clauses
        )
        
        # Combine results
        comprehensive_results = {
            **training_results,
            **coverage_metrics,
            'reduced_clauses': self.clause_reducer.convert_to_readable_dnf(reduced_clauses)
        }
        
        return comprehensive_results
    
    def visualize_results(self, results):
        """
        Create visualizations of model performance
        
        Args:
            results (dict): Performance results
        """
        # Performance Metrics Visualization
        metrics_to_plot = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'false_positive_rate', 'false_negative_rate'
        ]
        
        plt.figure(figsize=(12, 6))
        plt.bar(metrics_to_plot, [results.get(metric, 0) for metric in metrics_to_plot])
        plt.title('DNF Generator Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        plt.close()
        
        # Clause Complexity Visualization
        clause_lengths = [len(clause) for clause in results.get('reduced_clauses', [])]
        plt.figure(figsize=(10, 5))
        sns.histplot(clause_lengths, kde=True)
        plt.title('Distribution of Reduced DNF Clause Lengths')
        plt.xlabel('Clause Length')
        plt.ylabel('Frequency')
        plt.savefig('clause_length_distribution.png')
        plt.close()
    
    def run_comprehensive_experiment(self):
        """
        Run full experiment pipeline
        
        Returns:
            dict: Comprehensive experiment results
        """
        # Generate synthetic dataset
        train_dataset, val_dataset = self.generate_synthetic_dataset()
        
        # Train and evaluate
        results = self.train_and_evaluate(train_dataset, val_dataset)
        
        # Visualize results
        self.visualize_results(results)
        
        return results

# Main execution
if __name__ == "__main__":
    # Configuration
    config = {
        'num_variables': 10000,
        'embedding_dim': 8,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'fp_weight': 1.0,
        'fn_weight': 1.0
    }
    
    # Initialize and run experiment
    dnf_generator = DNFGeneratorIntegrator(config)
    experiment_results = dnf_generator.run_comprehensive_experiment()
    
    # Print key results
    print("Experiment Results:")
    print(f"Best F1 Score: {experiment_results.get('best_f1_score', 'N/A')}")
    print(f"Clause Coverage: {experiment_results.get('coverage', 'N/A')}")
    print(f"Number of Reduced Clauses: {len(experiment_results.get('reduced_clauses', []))}")
    
    # Save results to file
    import json
    with open('experiment_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)