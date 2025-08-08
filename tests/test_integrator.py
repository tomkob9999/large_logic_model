import unittest
from src.dnf_generator_integrator import DNFGeneratorIntegrator

class TestDNFGeneratorIntegrator(unittest.TestCase):
    def setUp(self):
        config = {
            'num_variables': 10000,
            'embedding_dim': 8
        }
        self.integrator = DNFGeneratorIntegrator(config)
    
    def test_initialization(self):
        self.assertEqual(self.integrator.config['num_variables'], 10000)
    
    def test_experiment_run(self):
        results = self.integrator.run_comprehensive_experiment()
        self.assertIsInstance(results, dict)

if __name__ == '__main__':
    unittest.main()