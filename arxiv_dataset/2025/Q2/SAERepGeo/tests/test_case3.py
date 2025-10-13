import unittest
import os
import pickle
import torch
import numpy as np
from unittest.mock import Mock, patch
from src.case3_intervention_analysis import run_intervention_analysis, CustomSAE
from src.sae_manifold_analyzer import SAEManifoldAnalyzer


class TestCase3(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.case1_output_dir = "test_outputs/case1_stratified_manifold"
        self.case2_output_dir = "test_outputs/case2_representation_structure"
        self.case3_output_dir = "test_outputs/case3_intervention_analysis"
        os.makedirs(self.case1_output_dir, exist_ok=True)
        os.makedirs(self.case2_output_dir, exist_ok=True)
        os.makedirs(self.case3_output_dir, exist_ok=True)

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

        # Mock concepts
        self.case2_3_concepts = ["test_concept"]

        # Mock tokens_dict
        self.tokens_dict = {
            "test_concept": {
                "TestModel": torch.ones((2, 5), dtype=torch.long)
            }
        }

        # Mock Case 1 and Case 2 caches
        self.results = {
            "test_concept": {
                "TestModel": {
                    0.0: {"latents": torch.ones((2, 5, 256))}
                }
            }
        }
        self.residual_cache = {
            "test_concept": {
                "TestModel": torch.ones((2, 5, 128))
            }
        }
        with open(os.path.join(self.case1_output_dir, 'results_zero_noise.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        with open(os.path.join(self.case1_output_dir, 'tokens_dict_zero_noise.pkl'), 'wb') as f:
            pickle.dump(self.tokens_dict, f)
        with open(os.path.join(self.case2_output_dir, 'residual_cache.pkl'), 'wb') as f:
            pickle.dump(self.residual_cache, f)

        # Mock clustering labels
        np.save(os.path.join(self.case2_output_dir, 'clusters_TestModel_test_concept_latents.npy'), np.zeros(10))

        # Mock model and SAE
        self.mock_model = Mock()
        self.mock_model.cfg.d_model = 128
        self.mock_model.tokenizer.convert_tokens_to_ids.return_value = 0
        self.mock_model.to_tokens.return_value = torch.ones((2, 5), dtype=torch.long)

        self.mock_sae = Mock(spec=CustomSAE)
        self.mock_sae.cfg.d_sae = 256
        self.mock_sae.encode.return_value = torch.ones((2, 5, 256))
        self.mock_sae.decode.return_value = torch.ones((2, 5, 128))
        self.mock_sae.W_dec = torch.ones((128, 256))
        self.mock_sae.b_dec = torch.zeros(128)

    @patch('src.case3_intervention_analysis.HookedSAETransformer.from_pretrained')
    @patch('src.case3_intervention_analysis.SAE.from_pretrained')
    def test_run_intervention_analysis(self, mock_sae_from_pretrained, mock_model_from_pretrained):
        mock_model_from_pretrained.return_value = self.mock_model
        mock_sae_from_pretrained.return_value = (self.mock_sae, None)

        analysis_results = run_intervention_analysis(
            self.model_configs, self.case2_3_concepts, self.case1_output_dir, self.case2_output_dir,
            self.case3_output_dir, self.tokens_dict, lambda_mse=1.0, device=self.device
        )

        # Check if output file exists
        results_file = os.path.join(self.case3_output_dir, 'intervention_analysis_results_icd_monge.csv')
        self.assertTrue(os.path.exists(results_file))

        # Check analysis_results structure
        self.assertIsInstance(analysis_results, list)
        if analysis_results:
            self.assertIn("concept", analysis_results[0])
            self.assertIn("model", analysis_results[0])
            self.assertIn("intervention", analysis_results[0])

    def tearDown(self):
        # Clean up test output directories
        for dir_path in [self.case1_output_dir, self.case2_output_dir, self.case3_output_dir]:
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
            os.rmdir(dir_path)
        os.rmdir("test_outputs")


if __name__ == '__main__':
    unittest.main()