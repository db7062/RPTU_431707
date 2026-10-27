# =============================================================================
# RUN SHAP FOR FILTERED AUTOSKLEARN PIPELINES
# =============================================================================

import os
import json
import pickle
import multiprocessing as mp

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from shap_rf_and_autosklearn import ClassificationAlgorithms
from Importance import FeatureImportance


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_IDS = [
    31,
    37,
    44,
    151,
    1461,
    4534
]

DATASET_PATHS = {
    31: "datasets/31",
    37: "datasets/37",
    44: "datasets/44",
    151: "datasets/151",
    1461: "datasets/1461",
    4534: "datasets/4534"
}

RANDOM_STATE = 42

RESULTS_DIR = "results"

BACKGROUND_SAMPLES = 50

N_JOBS = 6


# =============================================================================
# LOAD A SINGLE DATASET
# =============================================================================

def load_dataset(dataset_path):

    X = pd.read_csv(f"{dataset_path}/features.csv")

    y = pd.read_csv(f"{dataset_path}/targets.csv").iloc[:, 0]

    with open(f"{dataset_path}/categorical_indicator.json","r") as f:
        categorical_indicator = json.load(f)

    feature_names = list(X.columns)

    return X, y, feature_names, categorical_indicator


# =============================================================================
# LOAD BASELINE METRICS
# =============================================================================

def load_baseline_metrics(
    dataset_id,
    baseline_root=os.path.join(RESULTS_DIR, "baselines")
):

    metrics_file = os.path.join(
        baseline_root,
        str(dataset_id),
        "baseline_metrics.json"
    )

    if not os.path.exists(metrics_file):

        raise FileNotFoundError(metrics_file)

    with open(metrics_file,"r") as f:

        baseline_metrics = json.load(f)

    print("="*80)
    print(f"Dataset : {dataset_id}")
    print(f"Baseline Accuracy : {baseline_metrics['baseline_accuracy']:.4f}")
    print("="*80)

    return baseline_metrics


# =============================================================================
# LOAD FILTERED PIPELINES
# =============================================================================

# =============================================================================
# LOAD FILTERED PIPELINES
# =============================================================================

def load_filtered_pipelines(dataset_id):

    json_file = os.path.join(
        RESULTS_DIR,
        "filtered_pipelines",
        str(dataset_id),
        "filtered_pipelines.json"
    )

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Filtered pipeline file not found:\n{json_file}")

    with open(json_file, "r") as f:
        filtered_models = json.load(f)

    print(f"Dataset {dataset_id}: {len(filtered_models)} pipelines")

    return filtered_models


# =============================================================================
# LOAD ALL DATASETS
# =============================================================================

datasets = {}

for dataset_id in DATASET_IDS:

    dataset_path = DATASET_PATHS[dataset_id]

    X, y, feature_names, categorical_indicator = load_dataset(dataset_path)

    datasets[dataset_id] = {

        "dataset_id": dataset_id,

        "dataset_path": dataset_path,

        "X": X,

        "y": y,

        "feature_names": feature_names,

        "categorical_indicator": categorical_indicator

    }


# =============================================================================
# LOAD BASELINES
# =============================================================================

for dataset_id in DATASET_IDS:

    baseline = load_baseline_metrics(dataset_id)

    datasets[dataset_id]["baseline_accuracy"] = \
        baseline["baseline_accuracy"]

    datasets[dataset_id]["baseline_metrics"] = baseline



# =============================================================================
# RUN SHAP FOR ONE DATASET
# =============================================================================

def run_shap_for_dataset(dataset_id):

    print("=" * 80)
    print(f"Running SHAP for Dataset {dataset_id}")
    print("=" * 80)

    filtered_models = load_filtered_pipelines(dataset_id)

    dataset = datasets[dataset_id]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_cat,
        original_featureNames,
        original_catFeatureNames,
        shap_featureNames,
        shap_catfeatureNames,
        classes,
        nan_indices_
    ) = ClassificationAlgorithms.prepare_data_for_SHAP_importance(

        dataset["X"].copy(),
        dataset["y"].to_frame().copy(),
        dataset["dataset_path"],
        dataset["categorical_indicator"],
        RANDOM_STATE,
        datasetId=dataset_id
    )

    save_dir = os.path.join(
        RESULTS_DIR,
        "shap_values",
        str(dataset_id)
    )

    os.makedirs(save_dir, exist_ok=True)

    successful = 0
    failed = 0

    print(f"\nProcessing {len(filtered_models)} pipelines...\n")

    for i, pipeline_info in enumerate(filtered_models):

        model_path = pipeline_info["model_path"]

        config_id = os.path.splitext(
            os.path.basename(model_path)
        )[0]

        save_file = os.path.join(
            save_dir,
            config_id + ".json"
        )

        # Skip if already computed
        if os.path.exists(save_file):

            print(f"Skipping {config_id}")

            successful += 1

            continue

        print("=" * 80)
        print(f"Dataset {dataset_id}")
        print(f"Pipeline {i+1}/{len(filtered_models)}")
        print(config_id)
        print("=" * 80)

        try:

            with open(model_path, "rb") as fh:

                model = pickle.load(fh)

            y_hat = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_hat
            )

            (
                origFeature_abs_mean_shap,
                sum_one_class_shap,
                origFeature_abs_mean_shap_subcat,
                sum_one_class_shap_subcat

            ) = FeatureImportance.get_abs_SHAP_feature_importance(

                model,

                X_train,

                X_test,

                X_cat,

                original_featureNames,

                original_catFeatureNames,

                shap_featureNames,

                shap_catfeatureNames

            )

            featureImportances = {

                "dataset_id": dataset_id,

                "configPath": model_path,

                "config_id": config_id,

                "pipeline_configuration": str(model),

                "baseline_accuracy":
                    float(dataset["baseline_accuracy"]),

                "accuracy":
                    float(accuracy),

                "accuracy_difference":
                    float(pipeline_info["difference"]),

                "origFeature_abs_mean_shap":
                    origFeature_abs_mean_shap,

                "sum_one_class_shap":
                    sum_one_class_shap,

                "origFeature_abs_mean_shap_subcat":
                    origFeature_abs_mean_shap_subcat,

                "sum_one_class_shap_subcat":
                    sum_one_class_shap_subcat,

                "nan_indices":
                    nan_indices_.tolist()

            }

            with open(save_file, "w") as f:

                json.dump(
                    featureImportances,
                    f,
                    indent=4
                )

            successful += 1

            print(f"Saved : {config_id}")

        except Exception as e:

            failed += 1

            print(f"FAILED : {config_id}")

            print(e)

    print("\n")
    print("=" * 80)
    print(f"Dataset {dataset_id} Finished")
    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STARTING SHAP COMPUTATION")
    print("=" * 80)

    processes = []

    for dataset_id in DATASET_IDS:

        p = mp.Process(
            target=run_shap_for_dataset,
            args=(dataset_id,)
        )

        p.start()

        processes.append(p)

    for p in processes:

        p.join()

    print("\n")
    print("=" * 80)
    print("ALL DATASETS FINISHED")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    main()