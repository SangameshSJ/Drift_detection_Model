import pandas as pd
from drift_detection import DriftDetection
from plotting import generate_drift_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def main():
    # Load datasets
    # reference_data = pd.read_csv("cleaned_reference_data.csv")
    # current_data = pd.read_csv("cleaned_current_data.csv")

    # iris dataset
    # iris = load_iris()
    # iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # iris_df['species'] = iris.target
    # species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    # iris_df['species'] = iris_df['species'].map(species_map)
 
    # reference_df, current_df = train_test_split(iris_df, test_size=0.8, random_state=42)




    # reference_data = reference_df
    # current_data = current_df


    # differnt

    reference_data = pd.read_csv("/home/sigmoid/Downloads/refrence_d.csv")
    current_data = pd.read_csv("/home/sigmoid/Downloads/current_d.csv")


    # Prompt user to exclude specific columns
    print("Available columns:", list(reference_data.columns))
    excluded_columns = input(
        "Enter the columns to exclude from drift detection, separated by commas (or press Enter to skip): "
    ).split(",")
    excluded_columns = [col.strip() for col in excluded_columns if col.strip()]
    
    # Filter datasets to exclude specified columns
    reference_data = reference_data.drop(columns=excluded_columns, errors="ignore")
    current_data = current_data.drop(columns=excluded_columns, errors="ignore")

    # Initialize drift detector and perform drift detection
    drift_detector = DriftDetection(reference_data, current_data)
    drift_results = drift_detector.detect_drift()

    # Display results
    for column, result in drift_results.items():
        print(f"{column}: {result}")

    # Generate drift report
    generate_drift_report(reference_data, current_data, drift_results)

if __name__ == "__main__":
    main()