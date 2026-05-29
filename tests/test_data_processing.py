import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import build_processing_pipeline

def test_pipeline_output_shape():
    """Test if the pipeline returns the expected number of features."""
    df_mock = pd.DataFrame({
        'TransactionStartTime': ['2023-01-01 10:00:00'],
        'Amount': [1000.0],
        'Value': [1000.0],
        'ProviderId': ['P1'],
        'ProductId': ['PR1'],
        'ProductCategory': ['Cat1'],
        'ChannelId': ['C1'],
        'PricingStrategy': [1]
    })
    
    cat_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    num_features = ['Amount', 'Value']
    
    pipeline = build_processing_pipeline(cat_features, num_features)
    processed = pipeline.fit_transform(df_mock)
    
    # Check that it returns a 2D array (numpy or sparse)
    assert len(processed.shape) == 2
    assert processed.shape[0] == 1

def test_missing_value_handling():
    """Test if pipeline handles null values via imputation."""
    df_mock = pd.DataFrame({
        'TransactionStartTime': ['2023-01-01 10:00:00'],
        'Amount': [np.nan], # Missing value
        'Value': [1000.0],
        'ProviderId': ['P1'],
        'ProductId': ['PR1'],
        'ProductCategory': ['Cat1'],
        'ChannelId': ['C1'],
        'PricingStrategy': [1]
    })
    
    cat_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    num_features = ['Amount', 'Value']
    
    pipeline = build_processing_pipeline(cat_features, num_features)
    processed = pipeline.fit_transform(df_mock)
    
    assert not np.isnan(processed).any()