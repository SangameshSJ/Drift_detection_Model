import pandas as pd
from drift_detection import DriftDetection
from plotting import generate_drift_report
import custom_metrics  # Import the custom metrics

def main():
    # Load datasets
    # Load datasets
    reference_data = pd.read_csv("cleaned_reference_data.csv")
    current_data = pd.read_csv("cleaned_current_data.csv")

    #iris dataset
    # iris = load_iris()
    # iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # iris_df['species'] = iris.target
    # species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    # iris_df['species'] = iris_df['species'].map(species_map)
 
    # reference_df, current_df = train_test_split(iris_df, test_size=0.8, random_state=42)




    # reference_data = reference_df
    # current_data = current_df


    # differnt

    # reference_data = pd.read_csv("/home/sigmoid/Downloads/refrence_d.csv")
    # current_data = pd.read_csv("/home/sigmoid/Downloads/current_d.csv")
    
    # reference_data = pd.read_csv("cleaned_reference_data.csv")
    # current_data = pd.read_csv("cleaned_current_data.csv")

    # Prompt user to exclude specific columns
    print("Available columns:", list(reference_data.columns))
    excluded_columns = input(
        "Enter the columns to exclude from drift detection, separated by commas (or press Enter to skip): "
    ).split(",")
    excluded_columns = [col.strip() for col in excluded_columns if col.strip()]

    # Filter datasets to exclude specified columns
    reference_data = reference_data.drop(columns=excluded_columns, errors="ignore")
    current_data = current_data.drop(columns=excluded_columns, errors="ignore")

    # Perform drift detection with the default detection function
    custom1 = DriftDetection(reference_data, current_data)

    drift_results = custom1.detect_drift()

    # Display default drift detection results
    print("\nDrift Detection Results (default methods):")
    for column, result in drift_results.items():
        print(f"{column}: {result}")

    # List available custom metrics
    custom_features = {
        "mean_variance_shift": "Mean and Variance Shift",
        "cumulative_distribution_shift": "Cumulative Distribution Shift",
        "kullback_leibler_divergence": "Kullback-Leibler Divergence",
        "entropy_shift": "Entropy Shift",
        "outlier_ratio": "Outlier Ratio",
        "maximum_mean_discrepancy": "Maximum Mean Discrepancy",
        "proportion_unique_values": "Proportion of Unique Values",
        "jensen_shannon_divergence": "Jensen-Shannon Divergence"
    }

    # Prompt user to select custom features to compute
    print("\nAvailable Custom Features:")
    for key, feature in custom_features.items():
        print(f"{key}: {feature}")

    selected_features = input(
        "Enter the custom features to compute, separated by commas (or press Enter to use all): "
    ).split(",")
    selected_features = [feature.strip() for feature in selected_features if feature.strip()]

    # selected_features = [feature for feature in selected_features if feature in numerical_columns and feature in reference_data.columns and feature in current_data.columns]

    # Calculate selected custom metrics for drift detection
    print("\nCustom Metrics Results:")
    for feature in selected_features:
        if feature in custom_features:
            
            func = getattr(custom_metrics, feature)
            
            for column in reference_data.columns:
                if pd.api.types.is_numeric_dtype(reference_data[column]):
                    ref_data = reference_data[column]
                    curr_data = current_data[column]

                    try:
                        result = func(ref_data, curr_data)
                        print(f"{custom_features[feature]} for {column}: {result}")
                    except Exception as e:
                        print(f"Could not compute {custom_features[feature]} for {column} due to: {e}")
                    
                else:
                    pass
        else:
            print(f"Feature '{feature}' not found. Please check the available features.")

    # Generate drift report
    generate_drift_report(reference_data, current_data, drift_results)

if __name__ == "__main__":
    main()