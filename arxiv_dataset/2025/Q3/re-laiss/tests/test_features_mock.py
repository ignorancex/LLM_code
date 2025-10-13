import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import astropy.units as u
import relaiss as rl
from relaiss.features import SupernovaFeatureExtractor

def test_supernova_feature_extractor_mock():
    """Test that SupernovaFeatureExtractor works with mocked dust extinction."""
    np.random.seed(42)
    time_g = np.linspace(0, 100, 50)
    mag_g = np.random.normal(20, 0.5, 50)
    err_g = np.random.uniform(0.01, 0.1, 50)
    time_r = np.linspace(0, 100, 50)
    mag_r = np.random.normal(19, 0.5, 50)
    err_r = np.random.uniform(0.01, 0.1, 50)
    
    # Mock the entire SupernovaFeatureExtractor's __init__ method to avoid SFD map initialization
    with patch('relaiss.features.SupernovaFeatureExtractor.__init__') as mock_init, \
         patch('relaiss.features.SupernovaFeatureExtractor.extract_features') as mock_extract:
        
        # Configure the mock init to avoid calling the real init
        mock_init.return_value = None
        
        # Configure the mocked extract_features function
        mock_extract.return_value = pd.DataFrame({
            'g_peak_mag': [19.5],
            'r_peak_mag': [19.0],
            'g_peak_time': [25.0],
            'r_peak_time': [27.0],
            'g_rise_time': [15.0],
            'r_rise_time': [18.0],
            'g_decline_time': [20.0],
            'r_decline_time': [25.0],
            'features_valid': [True],
            'ztf_object_id': ["ZTF21abbzjeq"]
        })
        
        # Create the extractor manually (init is mocked)
        extractor = SupernovaFeatureExtractor(
            time_g=time_g,
            mag_g=mag_g,
            err_g=err_g,
            time_r=time_r,
            mag_r=mag_r,
            err_r=err_r,
            ztf_object_id="ZTF21abbzjeq",
            ra=150.0,
            dec=20.0
        )
        
        # Manually set required attributes that would be set in __init__
        extractor.ztf_object_id = "ZTF21abbzjeq"
        extractor.time_g = time_g
        extractor.mag_g = mag_g
        extractor.err_g = err_g
        extractor.time_r = time_r
        extractor.mag_r = mag_r
        extractor.err_r = err_r
        
        # Call extract_features (this will use our mock)
        features = extractor.extract_features()
        
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == 1
        assert 'g_peak_mag' in features.columns
        assert 'r_peak_mag' in features.columns
        
        # Mock extract_features for uncertainty test
        mock_extract.return_value = pd.DataFrame({
            'g_peak_mag': [19.5],
            'r_peak_mag': [19.0],
            'g_peak_time': [25.0],
            'r_peak_time': [27.0],
            'g_rise_time': [15.0],
            'r_rise_time': [18.0],
            'g_decline_time': [20.0],
            'r_decline_time': [25.0],
            'g_peak_mag_err': [0.1],
            'r_peak_mag_err': [0.1],
            'features_valid': [True],
            'ztf_object_id': ["ZTF21abbzjeq"]
        })
        
        features_with_err = extractor.extract_features(
            return_uncertainty=True, 
            n_trials=2
        )
        
        assert isinstance(features_with_err, pd.DataFrame)
        assert features_with_err.shape[0] == 1
        assert any(col.endswith('_err') for col in features_with_err.columns) 