import unittest
import os
import pickle
import torch
import numpy as np
from unittest.mock import Mock, patch
from src.case2_representation_structure import generate_case2_cache, analyze_representation_structure


class TestCase2(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.case1_output_dir = "test_outputs/case1_stratified_manifold"
        self.case2_output_dir = "test_outputs/case2_representation_structure"
        os.makedirs(self.case1_output_dir, exist_ok=True)
        os.makedirs(self.case2_output_dir, exist_ok=True)

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

        # Mock concept prompts and concepts
        self.concept_prompts = {
            "test_concept": ["test prompt 1", "test prompt 2"]
        }
        self.case2_3_concepts = ["test_concept"]

        # Mock Case 1 caches
        self.results = {
            "test_concept": {
                "TestModel": {
                    0.0: {"latents": torch.ones((2, 5, 256))}
                }
            }
        }
        self.tokens_dict = {
            "test_concept": {
                "TestModel": torch.ones((2, 5), dtype=torch.long)
            }
        }
        with open(os.path.join(self.case1_output_dir, 'results_zero_noise.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        with open(os.path.join(self.case1_output_dir, 'tokens_dict_zero_noise.pkl'), 'wb') as f:
            pickle.dump(self.tokens_dict, f)

        # Mock model and SAE
        self.mock_model = Mock()
        self.mock_model.cfg.d_model = 128
        self.mock_model.tokenizer.convert_tokens_to_ids.return_value = 0
        self.mock_model.to_tokens.return_value = torch.ones((2, 5), dtype=torch.long)

        self.mock_sae = Mock()
        self.mock_sae.cfg.d_sae = 256
        self.mock_sae.encode.return_value = torch.ones((2, 5, 256))

    @patch('src.case2_representation_structure.HookedSAETransformer.from_pretrained')
    @patch('src.case2_representation_structure.SAE.from_pretrained')
    def test_generate_case2_cache(self, mock_sae_from_pretrained, mock_model_from_pretrained):
        mock_model_from_pretrained.return_value = self.mock_model
        mock_sae_from_pretrained.return_value = (self.mock_sae, None)

        residual_cache = generate_case2_cache(
            self.model_configs, self.case2_3_concepts, self.concept_prompts, self.case2_output_dir, self.device
        )

        # Check if output files exist
        cache_file = os.path.join(self.case2_output_dir, 'residual_cache.pkl')
        resid_file = os.path.join(self.case2_output_dir, 'resid_TestModel_test_concept.pt')
        self.assertTrue(os.path.exists(cache_file))
        self.assertTrue(os.path.exists(resid_file))

        # Check residual_cache structure
        with open(cache_file, 'rb') as f:
            loaded_cache = pickle.load(f)
        self.assertIn("test_concept", loaded_cache)
        self.assertIn("TestModel", loaded_cache["test_concept"])

    def test_analyze_representation_structure(self):
        # Mock residual cache
        residual_cache = {
            "test_concept": {
                "TestModel": torch.ones((2, 5, 128))
            }
        }
        with open(os.path.join(self.case2_output_dir, 'residual_cache.pkl'), 'wb') as f:
            pickle.dump(residual_cache, f)

        analysis_results, loaded_cache = analyze_representation_structure(
            self.model_configs, self.case2_3_concepts, self.case1_output_dir, self.case2_output_dir
        )

        # Check if output files exist
        results_file = os.path.join(self.case2_output_dir, 'clustering_analysis_results.csv')
        self.assertTrue(os.path.exists(results_file))

        # Check analysis_results structure
        self.assertIsInstance(analysis_results, list)
        if analysis_results:
            self.assertIn("concept", analysis_results[0])
            self.assertIn("model", analysis_results[0])

    def tearDown(self):
        # Clean up test output directories
        for dir_path in [self.case1_output_dir, self.case2_output_dir]:
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
            os.rmdir(dir_path)
        os.rmdir("test_outputs")


if __name__ == '__main__':
    unittest.main()