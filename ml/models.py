from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ml.data_pipeline import (
    SEED,
    build_preprocessor,
    format_table,
    load_dataset,
    split_dataset,
)

import pandas as pd
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_model(model, output_path=Path("ml/best_model.joblib")):
    """
    Save the winning fitted pipeline to disk for reuse .
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path, compress=3)
    return output_path

def save_diagnostic_plot(model, X_test, y_test, output_path=Path("ml/q2_diagnostics.png")):
    """
    Generate and save residual and feature-importance diagnostic plots.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    regressor = model.named_steps["regressor"]
    importances = pd.Series(
        regressor.feature_importances_, index=X_test.columns
    ).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Residual
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=12, edgecolors="none")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Residuals on Test Set ({type(regressor).__name__})")
    axes[0].set_xlabel("Predicted Value (MedHouseVal)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].grid(alpha=0.25)

    # Feature-importance
    importances.plot(kind="barh", ax=axes[1], color="#2a7b88")
    axes[1].set_title("Feature Importances")
    axes[1].set_xlabel("Relative Importance (Impurity Reduction)")
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path



def build_baseline_models(random_state=SEED):
    """
    Create unifitted pipelines for the three baseline regression families.
    """

    estimators = {
        "Ridge": Ridge(alpha=1.0),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
    }

    return {
        name: Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model)])

        # Creates a unique pipeline for each model, including preprocessing and the regressor    
        for name, model in estimators.items()
    }

def evaluate_regressor(model, X, y):
    """
    Compute MSE, RMSE and MAE for a fitted regressor.
    """

    y_pred = model.predict(X)

    return {
        "mse": mean_squared_error(y, y_pred),
        "rmse": root_mean_squared_error(y, y_pred),
        "mae": mean_absolute_error(y, y_pred)
    }

def train_and_evaluate_baselines(sets, random_state=SEED):
    """
    Fit baseline models on train and evaluate on both train and validation.
    """

    models = build_baseline_models(random_state=random_state)
    rows = []

    for name, pipeline in models.items():
        pipeline.fit(sets["X_train"], sets["y_train"])

        train_metrics = evaluate_regressor(pipeline, sets["X_train"], sets["y_train"])
        val_metrics = evaluate_regressor(pipeline, sets["X_validation"], sets["y_validation"])

        rows.append({"model": name, "split": "train", **train_metrics})
        rows.append({"model": name, "split": "validation", **val_metrics})

    results_df = pd.DataFrame(rows).round(4)
    return models, results_df

def run_hyperparameter_study(sets, random_state=SEED):
    """
    Evaluate multiple RandomForest configurations on train and validation.
    """

    configs = [
        {"name": "RF_depth6_n100", "params": {"n_estimators": 100, "max_depth": 6}},
        {"name": "RF_depth12_n100", "params": {"n_estimators": 100, "max_depth": 12}},
        {"name": "RF_depth20_n100", "params": {"n_estimators": 100, "max_depth": 20}},
        {"name": "RF_depthNone_n200", "params": {"n_estimators": 200, "max_depth": None}},
    ]

    trained_models = {}
    rows = []

    for cfg in configs:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        **cfg["params"], random_state=random_state, n_jobs=-1
                    ),
                ),
            ]
        )
        pipeline.fit(sets["X_train"], sets["y_train"])
        trained_models[cfg["name"]] = pipeline

        train_metrics = evaluate_regressor(pipeline, sets["X_train"], sets["y_train"])
        val_metrics = evaluate_regressor(pipeline, sets["X_validation"], sets["y_validation"])

        rows.append({"config": cfg["name"], "split": "train", **train_metrics})
        rows.append({"config": cfg["name"], "split": "validation", **val_metrics})

    study_df = pd.DataFrame(rows).round(4)

    # Selects the best model based on the lowest RMSE on the validation set
    val_only = study_df[study_df["split"] == "validation"]
    best_name = val_only.loc[val_only["rmse"].idxmin(), "config"]

    return trained_models, study_df, best_name


def main():
    
    dataset = load_dataset()
    sets = split_dataset(dataset.data, dataset.target)

    _, baseline_results = train_and_evaluate_baselines(sets)
    print("=== Baseline Models (Train and Validation Metrics) ===")
    print(format_table(baseline_results, group_size=2))


    input("\n\nPress Enter to continue to hyperparameter study...\n\n")


    study_models, study_results, best_name = run_hyperparameter_study(sets)
    print("\n=== Hyperparameter Study (RandomForest) ===")
    print(format_table(study_results, group_size=2))
    print(f"\nBest configuration on validation: {best_name}")


    input("\n\nPress Enter to continue to final test evaluation...\n\n")


    best_model = study_models[best_name]
    test_metrics = evaluate_regressor(best_model, sets["X_test"], sets["y_test"])
    print("\n=== Final Test Evaluation (Winner Only) ===")
    print(format_table(pd.DataFrame([{"model": best_name, **test_metrics}])))


    input("\n\nPress Enter to save the winning model and diagnostic plots...\n\n")


    model_path = save_model(best_model)
    plot_path = save_diagnostic_plot(best_model, sets["X_test"], sets["y_test"])
    print(f"\nSaved model to: {model_path}")
    print(f"Saved plot to:  {plot_path}")



if __name__ == "__main__":
    main()