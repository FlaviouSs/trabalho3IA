from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

    return {
        "X_train": X_train,
        "X_validation": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_val,
        "y_test": y_test
    }

def main():

    # Fetch the California housing dataset
    x = fetch_california_housing(as_frame=True)

    sets = split_dataset(x.data, x.target)

    print("Training set shape:", sets["X_train"].shape)
    print("Validation set shape:", sets["X_validation"].shape)
    print("Test set shape:", sets["X_test"].shape)
    print("Total samples:", len(sets["X_train"]) + len(sets["X_validation"]) + len(sets["X_test"]))

if __name__ == "__main__":

    main()