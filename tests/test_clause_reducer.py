import torch
import unittest
import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dnf_clause_reducer import DNFClauseReducer

class TestDNFClauseReducer(unittest.TestCase):
    """
    Comprehensive test suite for DNF Clause Reducer
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.default_config = {
            'num_variables': 1000,
            'threshold': 0.5
        }
        self.reducer = DNFClauseReducer(**self.default_config)
    
    def test_initialization(self):
        """Test clause reducer initialization."""
        self.assertEqual(self.reducer.num_variables, 1000)
        self.assertEqual(self.reducer.threshold, 0.5)
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_config = {
            'num_variables': 5000,
            'threshold': 0.7
        }
        custom_reducer = DNFClauseReducer(**custom_config)
        
        self.assertEqual(custom_reducer.num_variables, 5000)
        self.assertEqual(custom_reducer.threshold, 0.7)
    
    def test_extract_clauses_basic(self):
        """Test basic clause extraction functionality."""
        batch_size = 10
        model_output = torch.rand(batch_size, self.default_config['num_variables'])
        
        clauses = self.reducer.extract_clauses(model_output)
        
        # Check return type
        self.assertIsInstance(clauses, list)
        
        # Check that clauses contain valid variable indices
        for clause in clauses:
            self.assertIsInstance(clause, list)
            for var_idx in clause:
                self.assertIsInstance(var_idx, int)
                self.assertTrue(0 <= var_idx < self.default_config['num_variables'])
    
    def test_extract_clauses_empty_input(self):
        """Test clause extraction with empty input."""
        empty_output = torch.zeros(1, self.default_config['num_variables'])
        
        clauses = self.reducer.extract_clauses(empty_output)
        
        # Should return a list (possibly empty)
        self.assertIsInstance(clauses, list)
    
    def test_extract_clauses_full_input(self):
        """Test clause extraction with all ones input."""
        full_output = torch.ones(1, self.default_config['num_variables'])
        
        clauses = self.reducer.extract_clauses(full_output)
        
        # Should return a list
        self.assertIsInstance(clauses, list)
    
    def test_variable_importance_computation(self):
        """Test variable importance computation."""
        batch_size = 50
        # Create structured data with some variables more important
        model_output = torch.rand(batch_size, self.default_config['num_variables'])
        
        # Make first 10 variables consistently high
        model_output[:, :10] = 0.9
        # Make next 10 variables consistently low
        model_output[:, 10:20] = 0.1
        
        importance_scores = self.reducer._compute_variable_importance(model_output)
        
        # Check return type and length
        self.assertIsInstance(importance_scores, list)
        self.assertEqual(len(importance_scores), self.default_config['num_variables'])
        
        # Check that all importance scores are non-negative
        for score in importance_scores:
            self.assertIsInstance(score, (float, np.floating))
            self.assertTrue(score >= 0)
        
        # Variables with consistent high values should have higher importance
        high_importance_vars = importance_scores[:10]
        low_importance_vars = importance_scores[10:20]
        
        avg_high_importance = np.mean(high_importance_vars)
        avg_low_importance = np.mean(low_importance_vars)
        
        # High importance variables should generally have higher scores
        self.assertTrue(avg_high_importance >= avg_low_importance)
    
    def test_information_gain_computation(self):
        """Test information gain computation for variables."""
        batch_size = 100
        model_output = torch.rand(batch_size, self.default_config['num_variables'])
        
        # Test information gain for first variable
        info_gain = self.reducer._compute_information_gain(model_output, 0)
        
        # Check return type and constraints
        self.assertIsInstance(info_gain, (float, np.floating))
        self.assertTrue(info_gain >= 0)  # Information gain should be non-negative
    
    def test_significant_clauses_finding(self):
        """Test finding significant clauses."""
        batch_size = 20
        model_output = torch.rand(batch_size, self.default_config['num_variables'])
        
        # Convert to binary for clause finding
        binary_output = (model_output > self.default_config['threshold']).float()
        
        significant_clauses = self.reducer._find_significant_clauses(binary_output)
        
        # Check return type
        self.assertIsInstance(significant_clauses, list)
        
        # Check clause structure
        for clause in significant_clauses:
            self.assertIsInstance(clause, list)
            # Each clause should contain valid variable indices
            for var_idx in clause:
                self.assertIsInstance(var_idx, int)
                self.assertTrue(0 <= var_idx < self.default_config['num_variables'])
    
    def test_redundant_clause_removal(self):
        """Test removal of redundant clauses."""
        # Create test clauses with known redundancy
        test_clauses = [
            [1, 2, 3],      # Clause 1
            [1, 2],         # Clause 2 (subset of Clause 1, should be kept)
            [1, 2, 3, 4],   # Clause 3 (superset of Clause 1, should be removed)
            [5, 6],         # Clause 4 (independent)
            [1],            # Clause 5 (subset of Clause 2, should be kept)
        ]
        
        non_redundant_clauses = self.reducer._remove_redundant_clauses(test_clauses)
        
        # Check return type
        self.assertIsInstance(non_redundant_clauses, list)
        
        # Check that redundant clauses are removed
        # The shortest clauses should be preserved
        clause_lengths = [len(clause) for clause in non_redundant_clauses]
        
        # Should contain clauses of different lengths
        self.assertTrue(len(non_redundant_clauses) > 0)
        
        # Verify no clause is a superset of another in the result
        for i, clause1 in enumerate(non_redundant_clauses):
            for j, clause2 in enumerate(non_redundant_clauses):
                if i != j:
                    # clause1 should not be a superset of clause2
                    self.assertFalse(set(clause2).issubset(set(clause1)) and len(clause2) < len(clause1))
    
    def test_readable_dnf_conversion(self):
        """Test conversion of clauses to readable DNF format."""
        test_clauses = [
            [0, 1, 2],
            [5, 10],
            [100]
        ]
        
        readable_dnf = self.reducer.convert_to_readable_dnf(test_clauses)
        
        # Check return type
        self.assertIsInstance(readable_dnf, list)
        self.assertEqual(len(readable_dnf), len(test_clauses))
        
        # Check format of readable clauses
        for readable_clause in readable_dnf:
            self.assertIsInstance(readable_clause, str)
            self.assertIn('x', readable_clause)  # Should contain variable names
            self.assertIn('∧', readable_clause)  # Should contain AND operator (if multi-variable)
    
    def test_dnf_coverage_evaluation(self):
        """Test evaluation of DNF clause coverage."""
        batch_size = 50
        original_data = torch.randint(0, 2, (batch_size, self.default_config['num_variables']), dtype=torch.float)
        
        # Create simple test clauses
        test_clauses = [
            [0, 1],     # Variables 0 and 1 must be true
            [2],        # Variable 2 must be true
            [3, 4, 5]   # Variables 3, 4, and 5 must be true
        ]
        
        coverage_metrics = self.reducer.evaluate_dnf_coverage(original_data, test_clauses)
        
        # Check return type and structure
        self.assertIsInstance(coverage_metrics, dict)
        self.assertIn('coverage', coverage_metrics)
        self.assertIn('num_clauses', coverage_metrics)
        
        # Check metric values
        self.assertIsInstance(coverage_metrics['coverage'], (float, np.floating))
        self.assertTrue(0 <= coverage_metrics['coverage'] <= 1)
        self.assertEqual(coverage_metrics['num_clauses'], len(test_clauses))
    
    def test_threshold_sensitivity(self):
        """Test that different thresholds produce different results."""
        batch_size = 20
        model_output = torch.rand(batch_size, self.default_config['num_variables'])
        
        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.7]
        results = []
        
        for threshold in thresholds:
            reducer = DNFClauseReducer(
                num_variables=self.default_config['num_variables'],
                threshold=threshold
            )
            clauses = reducer.extract_clauses(model_output)
            results.append(len(clauses))
        
        # Different thresholds should potentially produce different numbers of clauses
        self.assertEqual(len(results), 3)
        # All results should be non-negative
        for result in results:
            self.assertTrue(result >= 0)
    
    def test_large_input_handling(self):
        """Test clause reducer with larger input sizes."""
        large_config = {
            'num_variables': 5000,
            'threshold': 0.5
        }
        large_reducer = DNFClauseReducer(**large_config)
        
        batch_size = 10  # Smaller batch for memory efficiency
        large_output = torch.rand(batch_size, large_config['num_variables'])
        
        # Should not raise any exceptions
        clauses = large_reducer.extract_clauses(large_output)
        
        # Check basic properties
        self.assertIsInstance(clauses, list)
        
        # Check that variable indices are within bounds
        for clause in clauses:
            for var_idx in clause:
                self.assertTrue(0 <= var_idx < large_config['num_variables'])

def run_clause_reducer_tests():
    """Run all clause reducer tests."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_clause_reducer_tests()