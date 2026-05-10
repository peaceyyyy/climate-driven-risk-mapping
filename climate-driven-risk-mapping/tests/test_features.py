import pytest
import pandas as pd
import numpy as np
import os
from src.features.feature_engineering import engineer_features

def test_engineer_features(tmp_path):
    # Setup mock data
    input_file = tmp_path / "mock_joined.csv"
    output_file = tmp_path / "mock_engineered.csv"
    
    mock_data = {
        'siteID': ['A', 'A', 'B'],
        'year_month': ['2023-01', '2023-02', '2023-01'],
        'estimatedTotalCount': [10, 100, 0],
        'T2M': [10.0, 15.0, -5.0],
        'RH2M': [50.0, 60.0, 80.0],
        'soil_moisture': [0.2, 0.3, 0.1],
        'NDVI': [0.4, 0.5, 0.2]
    }
    pd.DataFrame(mock_data).to_csv(input_file, index=False)
    
    # Run engineering
    engineer_features(str(input_file), str(output_file))
    
    # Assert
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    
    # Check log transform
    # log1p(10) = ~2.397
    assert np.isclose(df.loc[0, 'log_density'], np.log1p(10))
    # log1p(0) = 0.0
    assert np.isclose(df.loc[2, 'log_density'], 0.0)
    
    # Check VPD existence
    assert 'VPD' in df.columns
    
    # Check lags
    assert 'T2M_lag30' in df.columns
    assert 'soil_moisture_roll90' in df.columns
