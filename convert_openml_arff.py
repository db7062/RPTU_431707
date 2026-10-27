import os
import json
import pandas as pd
from scipy.io import arff


def convert_openml_arff(arff_path, output_dir, target_column=None):
    """
    Convert an OpenML ARFF dataset into the format expected by the SHAP framework.

    Parameters
    ----------
    arff_path : str
        Path to the ARFF dataset.

    output_dir : str
        Folder where converted files will be stored.

    target_column : str or None
        Name of the target column.
        If None, the last column is assumed to be the target.
    """

    # --------------------------
    # Load ARFF
    # --------------------------
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)

    # Decode byte strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    # --------------------------
    # Identify target
    # --------------------------
    if target_column is None:
        target_column = df.columns[-1]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --------------------------
    # Detect categorical columns using ARFF metadata
    # --------------------------
    attribute_names = meta.names()[:-1]
    attribute_types = meta.types()[:-1]

    categorical_indicator = {
        name: (attr_type == "nominal")
        for name, attr_type in zip(attribute_names, attribute_types)
    }

    # --------------------------
    # Create output folder
    # --------------------------
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------
    # Save features
    # --------------------------
    X.to_csv(
        os.path.join(output_dir, "features.csv"),
        index=False
    )

    # --------------------------
    # Save targets
    # --------------------------
    y.to_csv(
        os.path.join(output_dir, "targets.csv"),
        index=False,
        header=True
    )

    # --------------------------
    # Save categorical indicators
    # --------------------------
    with open(
        os.path.join(output_dir, "categorical_indicator.json"),
        "w"
    ) as f:
        json.dump(categorical_indicator, f, indent=4)

    print("===================================")
    print("Conversion Complete")
    print("===================================")
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"Target column  : {target_column}")
    print(f"Output folder  : {output_dir}")


if __name__ == "__main__":

    convert_openml_arff(
        arff_path="dataset_31_credit-g.arff",
        output_dir="31",
        target_column="class"
    )