import unittest
import os
import pickle
import torch
from unittest.mock import Mock, patch
from src.case1_stratified_manifold import generate_case1_caches
from src.sae_manifold_analyzer import SAEManifoldAnalyzer


class TestCase1(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.output_dir = "test_outputs/case1_stratified_manifold"
        os.makedirs(self.output_dir, exist_ok=True)

        # Mock model configurations
        self.model_configs = {
            "TestModel": {
                "model_name": "test-model",
                "release": "test-release",
                "sae_id": "blocks.0.hook_resid_post",
                "hook": "blocks.0.hook_resid_post",
                "token": None,
                "d_model": 128
            }
        }

        # Mock concept prompts
        self.concept_prompts = {
            "test_concept": ["test prompt 1", "test prompt 2"]
        }

        # Mock model and SAE
        self.mock_model = Mock()
        self.mock_model.cfg.d_model = 128
        self.mock_model.tokenizer.convert_tokens_to_ids.return_value = 0
        self.mock_model.to_tokens.return_value = torch.ones((2, 5), dtype=torch.long)

        self.mock_sae = Mock()
        self.mock_sae.cfg.d_sae = 256
        self.mock_sae.encode.return_value = torch.ones((2, 5, 256))

    @patch('src.case1_stratified_manifold.HookedSAETransformer.from_pretrained')
    @patch('src.case1_stratified_manifold.SAE.from_pretrained')
    def test_generate_case1_caches(self, mock_sae_from_pretrained, mock_model_from_pretrained):
        mock_model_from_pretrained.return_value = self.mock_model
        mock_sae_from_pretrained.return_value = (self.mock_sae, None)

        results, tokens_dict = generate_case1_caches(
            self.model_configs, self.concept_prompts, self.output_dir, self.device
        )

        # Check if output files exist
        results_file = os.path.join(self.output_dir, 'results_zero_noise.pkl')
        tokens_file = os.path.join(self.output_dir, 'tokens_dict_zero_noise.pkl')
        self.assertTrue(os.path.exists(results_file))
        self.assertTrue(os.path.exists(tokens_file))

        # Check results structure
        with open(results_file, 'rb') as f:
            loaded_results = pickle.load(f)
        self.assertIn("test_concept", loaded_results)
        self.assertIn("TestModel", loaded_results["test_concept"])
        self.assertIn(0.0, loaded_results["test_concept"]["TestModel"])

        # Check tokens_dict structure
        with open(tokens_file, 'rb') as f:
            loaded_tokens = pickle.load(f)
        self.assertIn("test_concept", loaded_tokens)
        self.assertIn("TestModel", loaded_tokens["test_concept"])

    def tearDown(self):
        # Clean up test output directory
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))
        os.rmdir(self.output_dir)


if __name__ == '__main__':
    unittest.main()