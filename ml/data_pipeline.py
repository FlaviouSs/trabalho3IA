import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2221875


def load_dataset():
    """Load the California Housing dataset as pandas DataFrames."""
    return fetch_california_housing(as_frame=True)


def split_dataset(X, y, test_size=0.2, val_size=0.2, random_state=SEED):
    """Split features and target into disjoint train, validation, and test sets."""
    if test_size <= 0 or val_size <= 0:
        raise ValueError("test_size and val_size must be positive.")
    if test_size + val_size >= 1.0:
        raise ValueError("The sum of test_size and val_size must be less than 1.0.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=random_state, shuffle=True
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
        "y_test": y_test,
    }


def build_preprocessor():
    """Build an unfitted preprocessing pipeline with StandardScaler."""
    preprocessor = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor.set_output(transform="pandas")
    return preprocessor


def compute_summary(dataset, sets, X_train_scaled, X_val_scaled, X_test_scaled):
    """Compute Q1 cardinality, integrity, descriptive stats, and scaling checks."""
    df_raw = dataset.frame
    target_summary = pd.DataFrame({
        "train": sets["y_train"].describe(),
        "validation": sets["y_validation"].describe(),
        "test": sets["y_test"].describe(),
    }).T.round(4)

    scaling_check = pd.DataFrame({
        "train_mean": X_train_scaled.mean(),
        "train_std": X_train_scaled.std(ddof=0),
        "val_mean": X_val_scaled.mean(),
        "test_mean": X_test_scaled.mean(),
    }).round(4)

    return {
        "raw_shape": df_raw.shape,
        "split_shapes": {k: sets[k].shape for k in ("X_train", "X_validation", "X_test")},
        "missing_total": int(df_raw.isna().sum().sum()),
        "duplicated_rows": int(df_raw.duplicated().sum()),
        "train_stats": sets["X_train"].describe().T.round(4),
        "target_summary": target_summary,
        "scaling_check": scaling_check,
    }


def format_table(df, index_label=None, group_size=0):
    """Format a DataFrame as a padded ASCII table with column separators."""
    table_df = df.copy()
    if index_label is not None:
        table_df.insert(0, index_label, table_df.index.astype(str))

    headers = [str(col) for col in table_df.columns]
    is_numeric = [pd.api.types.is_numeric_dtype(table_df[col]) for col in table_df.columns]

    formatted_rows = []
    for _, row in table_df.iterrows():
        cells = []
        for col, numeric in zip(table_df.columns, is_numeric):
            val = row[col]
            if numeric:
                cells.append(f"{int(val)}" if col == "count" else f"{val:.4f}")
            else:
                cells.append(str(val))
        formatted_rows.append(cells)

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in formatted_rows))
        for i in range(len(headers))
    ]

    def render_row(cells):
        padded = [
            cells[i].rjust(widths[i]) if is_numeric[i] else cells[i].ljust(widths[i])
            for i in range(len(cells))
        ]
        return "  |  ".join(padded)

    divider = "--+--".join("-" * w for w in widths)
    lines = [render_row(headers), divider]
    for idx, row_cells in enumerate(formatted_rows):
        lines.append(render_row(row_cells))
        if group_size and (idx + 1) % group_size == 0 and (idx + 1) < len(formatted_rows):
            lines.append(divider)
    return "\n".join(lines)


def print_summary(summary):
    """Print a concise Q1 report from computed summary statistics."""
    s = summary["split_shapes"]
    print(
        f"Shapes    -> raw: {summary['raw_shape']} | "
        f"train: {s['X_train']} | val: {s['X_validation']} | test: {s['X_test']}"
    )
    print(
        f"Integrity -> missing: {summary['missing_total']} | "
        f"duplicated: {summary['duplicated_rows']}"
    )

    print("\n=== Train Feature Stats ===")
    print(format_table(summary["train_stats"], index_label="feature"))
    print("\n=== Target (MedHouseVal) Across Splits ===")
    print(format_table(summary["target_summary"], index_label="split"))
    print("\n=== Standardization Check (Fit on Train Only) ===")
    print(format_table(summary["scaling_check"], index_label="feature"))


def main():
    dataset = load_dataset()
    sets = split_dataset(dataset.data, dataset.target)

    preprocessor = build_preprocessor()

    # Fit the transformation pipeline on the training set (Calculates mean and std for scaling)
    X_train_scaled = preprocessor.fit_transform(sets["X_train"])

    # Apply the fitted transformation to the validation and test sets (Applies the same scaling as learned from the training set)
    X_val_scaled = preprocessor.transform(sets["X_validation"])
    X_test_scaled = preprocessor.transform(sets["X_test"])

    summary = compute_summary(dataset, sets, X_train_scaled, X_val_scaled, X_test_scaled)
    
    print_summary(summary)


def data_info():

    dataset = load_dataset()
    print(dataset.DESCR)

if __name__ == "__main__":
    main()
    #data_info()