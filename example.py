import torch
from src.dnf_generator_integrator import DNFGeneratorIntegrator

def main():
    """
    Demonstrate DNF Generator usage
    """
    # Configuration
    config = {
        'num_variables': 10000,
        'embedding_dim': 8,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5
    }
    
    # Initialize DNF Generator
    dnf_generator = DNFGeneratorIntegrator(config)
    
    # Generate synthetic dataset
    train_dataset, val_dataset = dnf_generator.generate_synthetic_dataset()
    
    # Train and evaluate
    results = dnf_generator.train_and_evaluate(train_dataset, val_dataset)
    
    # Print key results
    print("\n=== DNF Generator Experiment Results ===")
    print(f"Best F1 Score: {results.get('best_f1_score', 'N/A')}")
    print(f"Clause Coverage: {results.get('coverage', 'N/A')}")
    
    # Display some reduced clauses
    reduced_clauses = results.get('reduced_clauses', [])
    print("\nReduced DNF Clauses (first 5):")
    for clause in reduced_clauses[:5]:
        print(clause)

if __name__ == "__main__":
    main()