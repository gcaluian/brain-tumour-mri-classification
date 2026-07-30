# Random Forest successive-halving nested cross-validation
# Saves a checkpoint after every completed outer fold.

import json
import os
import time
from datetime import datetime
from pathlib import Path

# Prevent each parallel worker from starting additional numerical threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    HalvingRandomSearchCV,
    PredefinedSplit,
    StratifiedKFold
)


def save_csv_checkpoint(
    dataframe: pd.DataFrame,
    destination: Path
) -> None:
    """
    Save a DataFrame atomically.

    The temporary file is written first and renamed only after
    writing completes successfully.
    """
    temporary_file = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary_file,
        index=False
    )

    temporary_file.replace(destination)


def save_json_checkpoint(
    data: dict,
    destination: Path
) -> None:
    """Save a JSON file atomically."""
    temporary_file = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            default=str
        )

    temporary_file.replace(destination)


def main() -> None:
    # Resolving project paths from the location of this script
    project_root = Path(__file__).resolve().parents[1]

    features_file = (
        project_root
        / "extracted_features"
        / "hog_glcm_training_features.npz"
    )

    results_dir = (
        project_root
        / "results"
        / "hyperparameter_tuning"
        / "random_forest_successive_halving"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not features_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_file}"
        )

    # Loading the previously extracted HOG-GLCM features
    print("Loading feature archive...", flush=True)

    with np.load(
        features_file,
        allow_pickle=True
    ) as feature_archive:
        all_features = feature_archive[
            "features"
        ].astype(
            np.float32,
            copy=False
        )

        all_labels = feature_archive["labels"]

        all_folds = feature_archive[
            "folds"
        ].astype(int)

        if "relative_paths" in feature_archive.files:
            all_relative_paths = feature_archive[
                "relative_paths"
            ]
        else:
            all_relative_paths = np.arange(
                len(all_labels)
            ).astype(str)

    print(
        "Feature matrix shape:",
        all_features.shape,
        flush=True
    )

    print(
        "Labels shape:",
        all_labels.shape,
        flush=True
    )

    print(
        "Fold values:",
        sorted(np.unique(all_folds)),
        flush=True
    )

    expected_folds = [1, 2, 3, 4, 5]

    if sorted(np.unique(all_folds).tolist()) != expected_folds:
        raise ValueError(
            "Expected fold values 1-5."
        )

    # Using the existing outer-fold assignments
    predefined_cv_split = PredefinedSplit(
        test_fold=all_folds - 1
    )

    # Inner cross-validation used for parameter selection
    inner_cv_split = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    # n_estimators is excluded because successive halving
    # controls the number of trees.
    parameter_distributions = {
        "criterion": [
            "gini",
            "entropy"
        ],
        "max_depth": [
            None,
            20,
            40,
            60,
            80
        ],
        "min_samples_split": [
            2,
            5,
            10,
            20
        ],
        "min_samples_leaf": [
            1,
            2,
            4,
            8
        ],
        "max_features": [
            "sqrt",
            "log2",
            0.02,
            0.05,
            0.10,
            0.20
        ],
        "bootstrap": [
            True,
            False
        ]
    }

    random_forest_model = RandomForestClassifier(
        random_state=42,
        n_jobs=1
    )

    initial_candidates = 100
    halving_factor = 3
    minimum_trees = 25
    maximum_trees = 675

    # Saving the experiment configuration
    configuration = {
        "created_at": datetime.now().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "feature_file": str(features_file),
        "feature_matrix_shape": list(
            all_features.shape
        ),
        "initial_candidates": initial_candidates,
        "halving_factor": halving_factor,
        "minimum_trees": minimum_trees,
        "maximum_trees": maximum_trees,
        "inner_folds": 3,
        "scoring": "f1_macro",
        "search_parallel_jobs": 4,
        "random_state": 42,
        "parameter_distributions": (
            parameter_distributions
        )
    }

    save_json_checkpoint(
        configuration,
        results_dir / "run_configuration.json"
    )

    outer_splits = list(
        predefined_cv_split.split()
    )

    complete_run_start_time = (
        time.perf_counter()
    )

    for outer_fold_number, (
        outer_train_indices,
        outer_validation_indices
    ) in enumerate(
        outer_splits,
        start=1
    ):
        fold_summary_file = (
            results_dir
            / f"outer_fold_{outer_fold_number}_summary.csv"
        )

        fold_search_results_file = (
            results_dir
            / (
                f"outer_fold_{outer_fold_number}"
                "_search_results.csv"
            )
        )

        fold_predictions_file = (
            results_dir
            / (
                f"outer_fold_{outer_fold_number}"
                "_predictions.csv"
            )
        )

        # The summary file acts as the completion marker.
        if fold_summary_file.exists():
            print(
                f"\nOuter fold {outer_fold_number} "
                "already completed — skipping.",
                flush=True
            )
            continue

        print(
            f"\nStarting outer fold "
            f"{outer_fold_number}...",
            flush=True
        )

        X_outer_train = all_features[
            outer_train_indices
        ]

        X_outer_validation = all_features[
            outer_validation_indices
        ]

        y_outer_train = all_labels[
            outer_train_indices
        ]

        y_outer_validation = all_labels[
            outer_validation_indices
        ]

        halving_search = HalvingRandomSearchCV(
            estimator=random_forest_model,
            param_distributions=(
                parameter_distributions
            ),
            n_candidates=initial_candidates,
            factor=halving_factor,
            resource="n_estimators",
            min_resources=minimum_trees,
            max_resources=maximum_trees,
            scoring="f1_macro",
            cv=inner_cv_split,
            refit=True,
            random_state=42,
            n_jobs=4,
            verbose=1,
            return_train_score=False,
            error_score="raise"
        )

        fold_start_time = time.perf_counter()

        # Parameter tuning using only the outer-training data
        halving_search.fit(
            X_outer_train,
            y_outer_train
        )

        fold_tuning_time = (
            time.perf_counter()
            - fold_start_time
        )

        # Evaluation using the untouched outer-validation fold
        outer_validation_predictions = (
            halving_search.predict(
                X_outer_validation
            )
        )

        outer_validation_accuracy = (
            accuracy_score(
                y_outer_validation,
                outer_validation_predictions
            )
        )

        outer_validation_macro_f1 = f1_score(
            y_outer_validation,
            outer_validation_predictions,
            average="macro"
        )

        best_parameters = (
            halving_search.best_params_
        )

        # Saving all candidates and halving rounds
        fold_search_results = pd.DataFrame(
            halving_search.cv_results_
        )

        fold_search_results.insert(
            0,
            "outer_fold",
            outer_fold_number
        )

        save_csv_checkpoint(
            fold_search_results,
            fold_search_results_file
        )

        # Saving image-level outer-fold predictions
        fold_predictions = pd.DataFrame(
            {
                "outer_fold": outer_fold_number,
                "relative_path": (
                    all_relative_paths[
                        outer_validation_indices
                    ]
                ),
                "true_class": (
                    y_outer_validation
                ),
                "predicted_class": (
                    outer_validation_predictions
                )
            }
        )

        save_csv_checkpoint(
            fold_predictions,
            fold_predictions_file
        )

        # Saving the summary last, because it acts as
        # the completed-fold checkpoint.
        fold_summary = pd.DataFrame(
            [
                {
                    "fold": outer_fold_number,
                    "best_n_estimators": (
                        best_parameters[
                            "n_estimators"
                        ]
                    ),
                    "best_criterion": (
                        best_parameters[
                            "criterion"
                        ]
                    ),
                    "best_max_depth": (
                        best_parameters[
                            "max_depth"
                        ]
                    ),
                    "best_min_samples_split": (
                        best_parameters[
                            "min_samples_split"
                        ]
                    ),
                    "best_min_samples_leaf": (
                        best_parameters[
                            "min_samples_leaf"
                        ]
                    ),
                    "best_max_features": (
                        best_parameters[
                            "max_features"
                        ]
                    ),
                    "best_bootstrap": (
                        best_parameters[
                            "bootstrap"
                        ]
                    ),
                    "best_inner_macro_f1": (
                        halving_search.best_score_
                    ),
                    "outer_accuracy": (
                        outer_validation_accuracy
                    ),
                    "outer_macro_f1": (
                        outer_validation_macro_f1
                    ),
                    "tuning_time_minutes": (
                        fold_tuning_time / 60
                    )
                }
            ]
        )

        save_csv_checkpoint(
            fold_summary,
            fold_summary_file
        )

        print(
            f"\nOuter fold {outer_fold_number} "
            "complete.",
            flush=True
        )

        print(
            "Best parameters:",
            best_parameters,
            flush=True
        )

        print(
            "Best inner macro F1-score:",
            round(
                halving_search.best_score_,
                4
            ),
            flush=True
        )

        print(
            "Outer validation accuracy:",
            round(
                outer_validation_accuracy,
                4
            ),
            flush=True
        )

        print(
            "Outer validation macro F1-score:",
            round(
                outer_validation_macro_f1,
                4
            ),
            flush=True
        )

        print(
            "Tuning time:",
            round(
                fold_tuning_time / 60,
                2
            ),
            "minutes",
            flush=True
        )

        print(
            "Checkpoint saved:",
            fold_summary_file,
            flush=True
        )

    # Combining summaries for all completed folds
    completed_summary_files = sorted(
        results_dir.glob(
            "outer_fold_*_summary.csv"
        )
    )

    if not completed_summary_files:
        raise RuntimeError(
            "No completed fold summaries were found."
        )

    complete_results = pd.concat(
        [
            pd.read_csv(summary_file)
            for summary_file
            in completed_summary_files
        ],
        ignore_index=True
    ).sort_values("fold")

    save_csv_checkpoint(
        complete_results,
        results_dir
        / "combined_outer_fold_results.csv"
    )

    complete_run_time = (
        time.perf_counter()
        - complete_run_start_time
    )

    print(
        "\nCompleted successive-halving results:",
        flush=True
    )

    print(
        complete_results.to_string(index=False),
        flush=True
    )

    print(
        "\nCompleted outer folds:",
        len(complete_results),
        flush=True
    )

    print(
        "Mean outer validation accuracy:",
        round(
            complete_results[
                "outer_accuracy"
            ].mean(),
            4
        ),
        flush=True
    )

    print(
        "Outer accuracy standard deviation:",
        round(
            complete_results[
                "outer_accuracy"
            ].std(),
            4
        ),
        flush=True
    )

    print(
        "Mean outer validation macro F1-score:",
        round(
            complete_results[
                "outer_macro_f1"
            ].mean(),
            4
        ),
        flush=True
    )

    print(
        "Outer macro F1-score standard deviation:",
        round(
            complete_results[
                "outer_macro_f1"
            ].std(),
            4
        ),
        flush=True
    )

    print(
        "Runtime during this execution:",
        round(
            complete_run_time / 60,
            2
        ),
        "minutes",
        flush=True
    )


if __name__ == "__main__":
    main()