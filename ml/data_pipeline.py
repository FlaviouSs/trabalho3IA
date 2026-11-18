from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

SEED = 2221875


def print_dataset_info(dataset):
    """
    Prints the first 5 rows of the dataset, its shape, and a description.
    
    Parameters:
    dataset: The dataset object returned by fetch_california_housing.
    """
    print(dataset.DESCR, end="\n\n")
    print(f"Dataset shape: {dataset.data.shape}", end="\n\n")
    print(dataset.data.head(5), end="\n\n")
    print(f"Target: {dataset.target}")

def split_dataset(X, y, test_size=0.2, val_size=0.2, random_state=SEED):
    """
    Splits the dataset into training, validation, and test sets.
    
    Parameters:
    X: Features of the dataset.
    y: Target variable of the dataset.
    test_size: Proportion of the dataset to include in the test split (default is 0.2).
    val_size: Proportion of the dataset to include in the validation split (default is 0.2).
    random_state: Seed used by the random number generator (default is SEED).
    """

    if test_size <= 0 or val_size <= 0:
        raise ValueError("test_size and val_size parameters must be positive")

    if test_size + val_size >= 1.0:
        raise ValueError("The sum of test_size and val_size must be less than 1.0.")

    # Split the dataset into development and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, 
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Split the development set into training and validation sets
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative,
        random_state=random_state,
        shuffle=True
    )

    idx_train, idx_val, idx_test = set(X_train.index), set(X_val.index), set(X_test.index)
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)

    return {
        "X_train": X_train,
        "X_validation": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_val,
        "y_test": y_test
    }

def build_preprocessor():
    """
    Creates the preprocessing pipeline with standardization.
    """

    preprocessor = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    preprocessor.set_output(transform="pandas")
    return preprocessor

def summarize_dataset(dataset, sets):
    """
    Prints cardinality, dimensions, data integrity checks, and descriptive statistics.

    Parameters:
    dataset: The dataset object returned by fetch_california_housing.
    sets: Dictionary containing train, validation, and test splits.
    """
    df_raw = dataset.frame

    print("=== 1. Cardinality and Dimensions ===")
    print(f"Raw (features + target):\t\t\t\t{df_raw.shape}")
    print(f"Train (X):\t\t\t\t{sets['X_train'].shape} | y: {sets['y_train'].shape}")
    print(f"Validation (X):\t\t\t\t{sets['X_validation'].shape} | y: {sets['y_validation'].shape}")
    print(f"Test (X):\t\t\t\t{sets['X_test'].shape} | y: {sets['y_test'].shape}")

    print("\n=== 2. Raw Data Integrity ===")
    print("Missing values per column:")
    print(df_raw.isna().sum())
    print(f"Duplicated rows: {df_raw.duplicated().sum()}")

    print("\n=== 3. Feature Descriptive Statistics (Train Set) ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(sets["X_train"].describe().T.round(4))

    print("\n=== 4. Target Distribution (MedHouseVal) Across Splits ===")
    target_summary = pd.DataFrame({
        "train": sets["y_train"].describe(),
        "validation": sets["y_validation"].describe(),
        "test": sets["y_test"].describe(),
    }).T.round(4)
    print(target_summary)



def main():

    # Fetch the California housing dataset
    x = fetch_california_housing(as_frame=True)

    sets = split_dataset(x.data, x.target)

    preprocessor = build_preprocessor()

    X_train_scaled = preprocessor.fit_transform(sets["X_train"])
    X_val_scaled = preprocessor.transform(sets["X_validation"])
    X_test_scaled = preprocessor.transform(sets["X_test"])

    summarize_dataset(x, sets)

    print("\n=== 5. Standardization Check (Fit on Train Only) ===")
    print("Scaled shapes -> train:", X_train_scaled.shape, "| val:", X_val_scaled.shape, "| test:", X_test_scaled.shape)
    print("Train mean (expected ~0):\n", X_train_scaled.mean().round(4).to_dict())
    print("Validation mean (expected near 0, not exact):\n", X_val_scaled.mean().round(4).to_dict())

if __name__ == "__main__":

    main()