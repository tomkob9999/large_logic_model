import torch
import numpy as np
from itertools import combinations

class DNFClauseReducer:
    def __init__(self, num_variables=10000, threshold=0.5):
        """
        Initialize DNF Clause Reducer
        
        Args:
            num_variables (int): Total number of variables
            threshold (float): Probability threshold for clause inclusion
        """
        self.num_variables = num_variables
        self.threshold = threshold
    
    def extract_clauses(self, model_output):
        """
        Extract meaningful clauses from model output
        
        Args:
            model_output (torch.Tensor): Model's output tensor
        
        Returns:
            list: Reduced DNF clauses
        """
        # Convert model output to binary representation
        binary_output = (model_output > self.threshold).float()
        
        # Extract significant variables for each clause
        significant_clauses = self._find_significant_clauses(binary_output)
        
        # Reduce redundancy
        reduced_clauses = self._remove_redundant_clauses(significant_clauses)
        
        return reduced_clauses
    
    def _find_significant_clauses(self, binary_output):
        """
        Find significant clauses based on variable importance
        
        Args:
            binary_output (torch.Tensor): Binary representation of model output
        
        Returns:
            list: Significant clauses with their variables
        """
        significant_clauses = []
        
        # Compute variable importance
        variable_importance = self._compute_variable_importance(binary_output)
        
        # Sort variables by importance
        sorted_variables = sorted(
            range(len(variable_importance)), 
            key=lambda k: variable_importance[k], 
            reverse=True
        )
        
        # Select top variables to form clauses
        top_k_variables = sorted_variables[:min(100, len(sorted_variables))]
        
        # Generate clauses using top variables
        for clause_length in range(1, min(5, len(top_k_variables) + 1)):
            for clause_vars in combinations(top_k_variables, clause_length):
                clause = list(clause_vars)
                significant_clauses.append(clause)
        
        return significant_clauses
    
    def _compute_variable_importance(self, binary_output):
        """
        Compute importance of each variable
        
        Args:
            binary_output (torch.Tensor): Binary representation of model output
        
        Returns:
            list: Importance scores for each variable
        """
        # Compute frequency and impact of each variable
        variable_importance = []
        for var in range(self.num_variables):
            # Frequency of variable being 1
            frequency = torch.mean(binary_output[:, var]).item()
            
            # Compute information gain
            info_gain = self._compute_information_gain(binary_output, var)
            
            # Combined importance score
            importance = frequency * info_gain
            variable_importance.append(importance)
        
        return variable_importance
    
    def _compute_information_gain(self, binary_output, variable):
        """
        Compute information gain for a specific variable
        
        Args:
            binary_output (torch.Tensor): Binary representation of model output
            variable (int): Variable index
        
        Returns:
            float: Information gain score
        """
        # Split data based on variable
        var_positive = binary_output[binary_output[:, variable] == 1]
        var_negative = binary_output[binary_output[:, variable] == 0]
        
        # Compute entropy
        def compute_entropy(data):
            if len(data) == 0:
                return 0
            positive_ratio = torch.mean(data).item()
            negative_ratio = 1 - positive_ratio
            
            # Handle edge cases
            if positive_ratio == 0 or negative_ratio == 0:
                return 0
            
            entropy = -(
                positive_ratio * np.log2(positive_ratio) + 
                negative_ratio * np.log2(negative_ratio)
            )
            return entropy
        
        # Information gain calculation
        total_entropy = compute_entropy(binary_output)
        var_positive_entropy = compute_entropy(var_positive)
        var_negative_entropy = compute_entropy(var_negative)
        
        # Weighted entropy reduction
        info_gain = total_entropy - (
            (len(var_positive) / len(binary_output)) * var_positive_entropy +
            (len(var_negative) / len(binary_output)) * var_negative_entropy
        )
        
        return info_gain
    
    def _remove_redundant_clauses(self, clauses):
        """
        Remove redundant clauses
        
        Args:
            clauses (list): List of clauses
        
        Returns:
            list: Reduced set of non-redundant clauses
        """
        # Sort clauses by length (shorter clauses first)
        sorted_clauses = sorted(clauses, key=len)
        
        # Remove redundant clauses
        non_redundant_clauses = []
        for clause in sorted_clauses:
            # Check if current clause is already covered by existing clauses
            is_redundant = any(
                set(clause).issubset(set(existing_clause)) 
                for existing_clause in non_redundant_clauses
            )
            
            # Add clause if not redundant
            if not is_redundant:
                non_redundant_clauses.append(clause)
        
        return non_redundant_clauses
    
    def convert_to_readable_dnf(self, clauses):
        """
        Convert numeric clauses to readable DNF format
        
        Args:
            clauses (list): List of clause variable indices
        
        Returns:
            list: Readable DNF clauses
        """
        readable_dnf = []
        for clause in clauses:
            # Convert clause to readable format
            readable_clause = [f"x{var}" for var in clause]
            readable_dnf.append(" ∧ ".join(readable_clause))
        
        return readable_dnf
    
    def evaluate_dnf_coverage(self, original_data, reduced_clauses):
        """
        Evaluate the coverage and accuracy of reduced DNF clauses
        
        Args:
            original_data (torch.Tensor): Original input data
            reduced_clauses (list): Reduced DNF clauses
        
        Returns:
            dict: Coverage and accuracy metrics
        """
        # Convert clauses to boolean mask
        def clause_to_mask(clause, num_variables):
            mask = torch.zeros(num_variables, dtype=torch.bool)
            mask[list(clause)] = True
            return mask
        
        # Compute clause masks
        clause_masks = [
            clause_to_mask(clause, self.num_variables) 
            for clause in reduced_clauses
        ]
        
        # Compute coverage
        covered_samples = 0
        for sample in original_data:
            # Check if sample matches any clause
            sample_matches = any(
                torch.all(sample[mask] == 1) 
                for mask in clause_masks
            )
            if sample_matches:
                covered_samples += 1
        
        # Compute metrics
        coverage = covered_samples / len(original_data)
        
        return {
            'coverage': coverage,
            'num_clauses': len(reduced_clauses)
        }

# Example usage
if __name__ == "__main__":
    # Simulate large binary dataset
    num_variables = 10000
    num_samples = 100000
    
    # Generate random binary data
    data = torch.randint(0, 2, (num_samples, num_variables), dtype=torch.float)
    
    # Initialize DNF Clause Reducer
    dnf_reducer = DNFClauseReducer(num_variables=num_variables)
    
    # Extract clauses
    clauses = dnf_reducer.extract_clauses(data)
    
    # Convert to readable format
    readable_dnf = dnf_reducer.convert_to_readable_dnf(clauses)
    
    # Evaluate coverage
    coverage_metrics = dnf_reducer.evaluate_dnf_coverage(data, clauses)
    
    # Print results
    print("Reduced DNF Clauses:")
    for clause in readable_dnf[:10]:  # Print first 10 clauses
        print(clause)
    
    print("\nCoverage Metrics:")
    print(coverage_metrics)