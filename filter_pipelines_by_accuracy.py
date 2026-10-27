import os
import json

# ==========================================================
# SETTINGS
# ==========================================================

dataset_id = "31"

baseline_json = f"RF_shaps_100Samples/{dataset_id}/baseline_shap_{dataset_id}.json"

pipeline_folder = f"shap_values_autosklearn_configs_Xtest_50/{dataset_id}"

accuracy_threshold = 0.01

# ==========================================================
# LOAD BASELINE ACCURACY
# ==========================================================

with open(baseline_json, "r") as f:
    baseline = json.load(f)

baseline_accuracy = baseline["accuracy"]

print("=" * 80)
print(f"Dataset ID        : {dataset_id}")
print(f"Baseline Accuracy : {baseline_accuracy:.4f}")
print(f"Threshold         : ±{accuracy_threshold:.4f}")
print("=" * 80)

# ==========================================================
# READ ALL PIPELINES
# ==========================================================

pipeline_summary = []

all_json_files = sorted(
    [f for f in os.listdir(pipeline_folder) if f.endswith(".json")]
)

for file in all_json_files:

    with open(os.path.join(pipeline_folder, file), "r") as f:
        data = json.load(f)

    accuracy = data["accuracy"]
    difference = accuracy - baseline_accuracy

    selected = abs(difference) <= accuracy_threshold

    pipeline_summary.append({
        "config_id": data["config_id"],
        "accuracy": accuracy,
        "difference": difference,
        "selected": selected,
    })

# ==========================================================
# SUMMARY
# ==========================================================

selected_count = sum(p["selected"] for p in pipeline_summary)

print(f"\nTotal Pipelines      : {len(pipeline_summary)}")
print(f"Selected Pipelines   : {selected_count}")

print("\n")
print("=" * 110)
print(f"{'Config ID':20} {'Accuracy':>10} {'Diff':>10} {'Selected':>10}")
print("=" * 110)

for pipeline in sorted(
    pipeline_summary,
    key=lambda x: x["accuracy"],
    reverse=True
):

    print(
        f"{pipeline['config_id']:20}"
        f"{pipeline['accuracy']:>10.4f}"
        f"{pipeline['difference']:>10.4f}"
        f"{'YES' if pipeline['selected'] else 'NO':>10}"
    )

# ==========================================================
# DETAILED INFORMATION FOR SELECTED PIPELINES
# ==========================================================

print("\n\n")
print("=" * 110)
print("SELECTED PIPELINES")
print("=" * 110)

for pipeline in pipeline_summary:

    if not pipeline["selected"]:
        continue

    print("\n" + "-" * 110)

    print(f"Pipeline ID : {pipeline['config_id']}")
    print(f"Accuracy    : {pipeline['accuracy']:.4f}")
    print(f"Difference  : {pipeline['difference']:+.4f}")


print("\n")
print("=" * 110)
print("Analysis Complete")
print("=" * 110)