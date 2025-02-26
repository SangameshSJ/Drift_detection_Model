from algorithms import (
    run_anderson_darling_normality,
    run_ks_test,
    run_wasserstein_distance,
    run_js_divergence,
    run_chi_squared_test,
    run_ttest
)
import pandas as pd
import numpy as np

def detect_drift(reference_data, current_data):
    """
    Detects data drift using various statistical tests for numerical and categorical columns.

    Parameters:
        reference_data (pd.DataFrame): Historical dataset for comparison.
        current_data (pd.DataFrame): Current dataset to be analyzed for drift.

    Returns:
        dict: A dictionary with drift metrics for each column.
    """
    results = {}
    threshold = 0.1  # Default threshold for large datasets

    # Classify columns based on data types
    column_mapping = {
        'numerical_features': [],
        'categorical_features': []
    }

    for column in reference_data.columns:
        if np.issubdtype(reference_data[column].dtype, np.number):
            column_mapping['numerical_features'].append(column)
        else:
            column_mapping['categorical_features'].append(column)

    for column in column_mapping['numerical_features']:
        n_unique = reference_data[column].nunique()
        
        if len(reference_data) > 1000 or len(current_data) > 1000:
            # Use Wasserstein Distance for large datasets
            drift_metric, test_name = run_wasserstein_distance(reference_data[column], current_data[column])
        elif n_unique > 5:  # Numerical column with > 5 unique values
            try:
                # Check if data is normal using Anderson-Darling
                ad_stat, is_normal = run_anderson_darling_normality(reference_data[column], current_data[column])
                if is_normal:
                    # Use T-test for normal data
                    drift_metric, test_name = run_ttest(reference_data[column], current_data[column])
                else:
                    # Use KS Test for non-normal data
                    drift_metric, test_name = run_ks_test(reference_data[column], current_data[column])
            except:
                # Fallback to Wasserstein Distance
                drift_metric, test_name = run_wasserstein_distance(reference_data[column], current_data[column])
        else:
            # Use JS Divergence for fewer unique values
            drift_metric, test_name = run_js_divergence(reference_data[column], current_data[column])

        results[column] = {
            "Test Name": test_name,
            "Drift Metric": drift_metric,
            "Threshold Breach": drift_metric > threshold
        }

    for column in column_mapping['categorical_features']:
        try:
            # Use Chi-Squared Test for categorical features
            drift_metric, test_name = run_chi_squared_test(reference_data[column], current_data[column])
        except:
            # Fallback to JS Divergence for categorical features
            drift_metric, test_name = run_js_divergence(reference_data[column], current_data[column])

        results[column] = {
            "Test Name": test_name,
            "Drift Metric": drift_metric,
            "Threshold Breach": drift_metric > threshold
        }

    return results
