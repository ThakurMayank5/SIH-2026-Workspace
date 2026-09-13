"""
VAANI MODEL V1
Threshold Evaluation

Evaluates the trained DS-CNN at different Vaani probability
thresholds.

IMPORTANT:
The preprocessing here MUST match train_dscnn.py exactly.

Classes:
    0 = Negative
    1 = Vaani

Input features:
    40 x 49 log-Mel spectrogram

Run from project root:

    python model_v1/scripts/evaluate_thresholds.py
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "model_v1" / "features"
MODEL_PATH = (
    PROJECT_ROOT /
    "model_v1" /
    "exports" /
    "vaani_dscnn_v1.keras"
)

EVALUATION_DIR = (
    PROJECT_ROOT /
    "model_v1" /
    "evaluation"
)

OUTPUT_PATH = (
    EVALUATION_DIR /
    "threshold_metrics.json"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Thresholds to evaluate
THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.99
]


# ============================================================
# LOAD TEST FEATURES
# ============================================================

def load_test_features():

    path = FEATURE_DIR / "test.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Test feature file not found:\n{path}"
        )

    data = np.load(path)

    X_test = data["X"]
    y_test = data["y"]

    print()
    print("Test features:")
    print(f"X shape: {X_test.shape}")
    print(f"y shape: {y_test.shape}")

    print()
    print("Labels:")
    print(
        f"Negative (0): "
        f"{np.sum(y_test == 0)}"
    )
    print(
        f"Vaani (1):    "
        f"{np.sum(y_test == 1)}"
    )

    return X_test, y_test


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(X_test):

    print()
    print("=" * 60)
    print("PREPARING TEST FEATURES")
    print("=" * 60)

    print()
    print("Raw feature range:")
    print(
        f"{X_test.min():.3f} -> "
        f"{X_test.max():.3f}"
    )

    # --------------------------------------------------------
    # Add channel dimension
    #
    # (N, 40, 49)
    #       ↓
    # (N, 40, 49, 1)
    # --------------------------------------------------------

    if X_test.ndim == 3:

        X_test = X_test[..., np.newaxis]

    elif X_test.ndim != 4:

        raise ValueError(
            f"Unexpected feature shape: "
            f"{X_test.shape}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This MUST exactly match train_dscnn.py.
    #
    # Raw log-Mel:
    #
    #       -80 ... 0 dB
    #
    # Normalized:
    #
    #        0 ... 1
    # --------------------------------------------------------

    X_test = (
        X_test + 80.0
    ) / 80.0

    X_test = np.clip(
        X_test,
        0.0,
        1.0
    )

    print()
    print("Prepared feature shape:")
    print(X_test.shape)

    print()
    print("Prepared feature range:")
    print(
        f"{X_test.min():.3f} -> "
        f"{X_test.max():.3f}"
    )

    return X_test


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    vaani_probability,
    threshold
):

    # --------------------------------------------------------
    # Threshold decision
    #
    # If P(Vaani) >= threshold:
    #       predict Vaani
    #
    # Otherwise:
    #       predict Negative
    # --------------------------------------------------------

    y_pred = (
        vaani_probability >= threshold
    ).astype(np.int32)

    # --------------------------------------------------------
    # Confusion matrix components
    # --------------------------------------------------------

    tn = int(
        np.sum(
            (y_true == 0) &
            (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0) &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1) &
            (y_pred == 0)
        )
    )

    tp = int(
        np.sum(
            (y_true == 1) &
            (y_pred == 1)
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = (
        (tp + tn) /
        len(y_true)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {

        "threshold": float(threshold),

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall_tpr": float(recall),

        "f1": float(f1),

        "fpr": float(false_positive_rate),

        "fnr": float(false_negative_rate),

        "tp": tp,

        "tn": tn,

        "fp": fp,

        "fn": fn
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()
    print("=" * 110)
    print("THRESHOLD RESULTS")
    print("=" * 110)

    print()

    print(
        f"{'Threshold':>10} "
        f"{'Accuracy':>10} "
        f"{'Precision':>11} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FPR':>10} "
        f"{'FNR':>10} "
        f"{'TP':>5} "
        f"{'FP':>5} "
        f"{'FN':>5} "
        f"{'TN':>5}"
    )

    print("-" * 110)

    for result in results:

        print(
            f"{result['threshold']:>10.2f} "
            f"{result['accuracy'] * 100:>9.2f}% "
            f"{result['precision'] * 100:>10.2f}% "
            f"{result['recall_tpr'] * 100:>9.2f}% "
            f"{result['f1'] * 100:>9.2f}% "
            f"{result['fpr'] * 100:>9.2f}% "
            f"{result['fnr'] * 100:>9.2f}% "
            f"{result['tp']:>5} "
            f"{result['fp']:>5} "
            f"{result['fn']:>5} "
            f"{result['tn']:>5}"
        )

    print("-" * 110)


# ============================================================
# FIND RECOMMENDED THRESHOLD
# ============================================================

def find_recommended_threshold(results):

    # --------------------------------------------------------
    # Engineering requirement for this evaluation:
    #
    # First preserve at least 95% recall.
    #
    # Among those thresholds, choose the one with the
    # lowest FPR.
    # --------------------------------------------------------

    eligible = [

        result

        for result in results

        if result["recall_tpr"] >= 0.95
    ]

    print()
    print("=" * 60)
    print("THRESHOLD SELECTION")
    print("=" * 60)

    if not eligible:

        print()
        print(
            "No tested threshold maintains "
            "at least 95% recall."
        )

        return None

    recommended = min(
        eligible,
        key=lambda result: (
            result["fpr"],
            -result["recall_tpr"]
        )
    )

    print()
    print(
        "Best threshold with Recall >= 95%:"
    )

    print()
    print(
        f"Threshold : "
        f"{recommended['threshold']:.2f}"
    )

    print(
        f"Recall    : "
        f"{recommended['recall_tpr'] * 100:.2f}%"
    )

    print(
        f"FPR       : "
        f"{recommended['fpr'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{recommended['precision'] * 100:.2f}%"
    )

    print(
        f"F1        : "
        f"{recommended['f1'] * 100:.2f}%"
    )

    print()
    print("Confusion matrix at this threshold:")

    print()
    print(
        f"TP = {recommended['tp']}"
    )

    print(
        f"FP = {recommended['fp']}"
    )

    print(
        f"FN = {recommended['fn']}"
    )

    print(
        f"TN = {recommended['tn']}"
    )

    return recommended


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    recommended
):

    output = {

        "model": str(MODEL_PATH),

        "classes": {
            "0": "negative",
            "1": "vaani"
        },

        "preprocessing": {
            "input_range_before_normalization": [
                -80.0,
                0.0
            ],
            "normalization": "(x + 80.0) / 80.0",
            "input_range_after_normalization": [
                0.0,
                1.0
            ]
        },

        "thresholds_evaluated": results,

        "selection_rule":
            "Lowest FPR among thresholds "
            "with recall >= 95%",

        "recommended_threshold": (
            recommended["threshold"]
            if recommended is not None
            else None
        )
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print()
    print(
        f"Saved results:\n{OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("VAANI DS-CNN V1 - THRESHOLD EVALUATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_test, y_test = (
        load_test_features()
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X_test = prepare_test_data(
        X_test
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    print()
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)

    probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=0
    )

    # --------------------------------------------------------
    # Validate model output
    # --------------------------------------------------------

    if probabilities.ndim != 2:

        raise ValueError(
            "Unexpected model output shape: "
            f"{probabilities.shape}"
        )

    if probabilities.shape[1] != 2:

        raise ValueError(
            "Expected two output classes. "
            f"Got: {probabilities.shape}"
        )

    # --------------------------------------------------------
    # Class mapping from train_dscnn.py:
    #
    # 0 = Negative
    # 1 = Vaani
    #
    # Therefore Vaani probability is index 1.
    # --------------------------------------------------------

    vaani_probability = (
        probabilities[:, 1]
    )

    print()
    print("Model output check:")

    print()
    print(
        f"Vaani probability min: "
        f"{vaani_probability.min():.6f}"
    )

    print(
        f"Vaani probability max: "
        f"{vaani_probability.max():.6f}"
    )

    print(
        f"Vaani probability mean: "
        f"{vaani_probability.mean():.6f}"
    )

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    results = []

    for threshold in THRESHOLDS:

        result = calculate_metrics(
            y_true=y_test,
            vaani_probability=vaani_probability,
            threshold=threshold
        )

        results.append(result)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # Select threshold
    # --------------------------------------------------------

    recommended = (
        find_recommended_threshold(
            results
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        recommended
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("THRESHOLD EVALUATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":

    main()