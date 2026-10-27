

import os

# Limit numerical-library threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import json
import time
import pickle
import traceback

import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed

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
    1461,
    4534
]

DATASET_PATHS = {
    31: "datasets/31",
    37: "datasets/37",
    44: "datasets/44",
    1461: "datasets/1461",
    4534: "datasets/4534"
}

RESULTS_DIR = "results"

RANDOM_STATE = 42

# Number of parallel workers
N_WORKERS = 32

def load_dataset(dataset_path):

    X = pd.read_csv(
        os.path.join(dataset_path, "features.csv")
    )

    y = pd.read_csv(
        os.path.join(dataset_path, "targets.csv")
    ).iloc[:, 0]

    with open(
        os.path.join(dataset_path, "categorical_indicator.json"),
        "r"
    ) as f:

        categorical_indicator = json.load(f)

    return X, y, categorical_indicator

def load_baseline(dataset_id):

    baseline_file = os.path.join(
        RESULTS_DIR,
        "baselines",
        str(dataset_id),
        "baseline_metrics.json"
    )

    with open(baseline_file) as f:

        return json.load(f)


def load_filtered_pipelines(dataset_id):

    file = os.path.join(
        RESULTS_DIR,
        "filtered_pipelines",
        str(dataset_id),
        "filtered_pipelines.json"
    )

    with open(file) as f:

        return json.load(f)



# =============================================================================
# PREPARE DATASET FOR SHAP
# =============================================================================

def prepare_dataset(dataset_id):

    dataset_path = DATASET_PATHS[dataset_id]

    print("=" * 80)
    print(f"Preparing Dataset {dataset_id}")
    print("=" * 80)

    # ------------------------------------------------------------
    # Load raw dataset
    # ------------------------------------------------------------

    X, y, categorical_indicator = load_dataset(
        dataset_path
    )

    # ------------------------------------------------------------
    # Load baseline
    # ------------------------------------------------------------

    baseline = load_baseline(
        dataset_id
    )

    baseline_accuracy = baseline["baseline_accuracy"]

    # ------------------------------------------------------------
    # Prepare SHAP data
    # ------------------------------------------------------------

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

        X.copy(),

        y.to_frame().copy(),

        dataset_path,

        categorical_indicator,

        RANDOM_STATE,

        datasetId=dataset_id

    )

    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------

    save_dir = os.path.join(
        RESULTS_DIR,
        "shap_values",
        str(dataset_id)
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Load filtered pipelines
    # ------------------------------------------------------------

    filtered_models = load_filtered_pipelines(
        dataset_id
    )

    print(f"Filtered Pipelines : {len(filtered_models)}")
    print(f"Baseline Accuracy  : {baseline_accuracy:.4f}")

    return {

        "dataset_id": dataset_id,

        "baseline_accuracy": baseline_accuracy,

        "filtered_models": filtered_models,

        "save_dir": save_dir,

        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,

        "X_cat": X_cat,

        "original_featureNames": original_featureNames,

        "original_catFeatureNames": original_catFeatureNames,

        "shap_featureNames": shap_featureNames,

        "shap_catfeatureNames": shap_catfeatureNames,

        "nan_indices": nan_indices_

    }

# =============================================================================
# WORKER GLOBALS
# =============================================================================

DATA = None


def init_worker(prepared_dataset):

    global DATA

    DATA = prepared_dataset


# =============================================================================
# PROCESS ONE PIPELINE
# =============================================================================

def process_pipeline(pipeline_info):



    print(f"Worker started: {os.getpid()}")

    global DATA

   

    dataset_id = DATA["dataset_id"]

   

    save_dir = DATA["save_dir"]

    model_path = pipeline_info["model_path"]



    pipeline_name = os.path.basename(
    os.path.dirname(model_path)
)

    model_name = os.path.splitext(
        os.path.basename(model_path)
    )[0]

    config_id = f"{pipeline_name}_{model_name}"

    save_file = os.path.join(
        save_dir,
        config_id + ".json"
    )

    # --------------------------------------------------------
    # Skip existing
    # --------------------------------------------------------

    if os.path.exists(save_file):

        return (
            config_id,
            "SKIPPED"
        )

    try:

        with open(model_path, "rb") as fh:

            model = pickle.load(fh)

        print("5. Model Loaded")

        y_hat = model.predict(DATA["X_test"])

        print("6. Prediction finished")

        accuracy = accuracy_score(
            DATA["y_test"],
            y_hat
        )

        (
            origFeature_abs_mean_shap,
            sum_one_class_shap,
            origFeature_abs_mean_shap_subcat,
            sum_one_class_shap_subcat

        ) = FeatureImportance.get_abs_SHAP_feature_importance(

            model,

            DATA["X_train"],

            DATA["X_test"],

            DATA["X_cat"],

            DATA["original_featureNames"],

            DATA["original_catFeatureNames"],

            DATA["shap_featureNames"],

            DATA["shap_catfeatureNames"]

        )

        print("7. SHAP finished")

        featureImportances = {

            "dataset_id": dataset_id,

            "configPath": model_path,

            "config_id": config_id,

            "pipeline_configuration": str(model),

            "baseline_accuracy":
                float(DATA["baseline_accuracy"]),

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
                DATA["nan_indices"].tolist()

        }

        with open(save_file, "w") as f:

            json.dump(
                featureImportances,
                f,
                indent=4
            )

        return (

            config_id,

            "SUCCESS"

        )

    except Exception:

        return (

            config_id,

            "FAILED",

            traceback.format_exc()

        )


# =============================================================================
# RUN DATASET IN PARALLEL
# =============================================================================

def run_dataset(dataset_id):

    data = prepare_dataset(dataset_id)

    print("\n")
    print("=" * 80)
    print(f"Starting Dataset {dataset_id}")
    print(f"Pipelines : {len(data['filtered_models'])}")
    print(f"Workers   : {N_WORKERS}")
    print("=" * 80)

    completed = 0
    successful = 0
    skipped = 0
    failed = 0

    start = time.time()

    ctx = mp.get_context("fork")

    with ProcessPoolExecutor(

        max_workers=N_WORKERS,

        mp_context=ctx,

        initializer=init_worker,

        initargs=(data,)

    ) as executor:

        futures = {

            executor.submit(
                process_pipeline,
                pipeline
            ): pipeline

            for pipeline in data["filtered_models"]

        }

        for future in as_completed(futures):

            

            result = future.result()

            completed += 1

            status = result[1]

            if status == "SUCCESS":

                successful += 1

            elif status == "SKIPPED":

                skipped += 1

            else:

                failed += 1

                print("\n")
                print("=" * 80)
                print(result[0])
                print(result[2])
                print("=" * 80)

            elapsed = time.time() - start

            rate = completed / elapsed

            remaining = len(data["filtered_models"]) - completed

            eta = remaining / rate if rate > 0 else 0

            print(

                f"\r"

                f"{completed}/{len(data['filtered_models'])}"

                f" | Success {successful}"

                f" | Skip {skipped}"

                f" | Failed {failed}"

                f" | ETA {eta/3600:.2f} h",

                end=""

            )

    print()

    print("=" * 80)
    print(f"Finished Dataset {dataset_id}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("PARALLEL SHAP COMPUTATION")
    print("=" * 80)

    for dataset_id in DATASET_IDS:

        run_dataset(dataset_id)

    print("\n")
    print("=" * 80)
    print("ALL DATASETS FINISHED")
    print("=" * 80)


if __name__ == "__main__":

    mp.set_start_method("fork")

    main()