from pathlib import Path
import json

import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "model_v1"
    / "checkpoints"
    / "best_model.keras"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "model_v1"
    / "features"
)

TRAIN_FEATURE_PATH = FEATURE_DIR / "train.npz"
TEST_FEATURE_PATH = FEATURE_DIR / "test.npz"

EXPORT_DIR = (
    PROJECT_ROOT
    / "model_v1"
    / "exports"
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

INPUT_SHAPE = (40, 49, 1)

# Number of samples used to calibrate int8 ranges.
CALIBRATION_SAMPLES = 300

# Fixed seed so the calibration subset is reproducible.
CALIBRATION_SEED = 42

# Decision threshold on P(Vaani) used at deployment — must
# match whatever threshold the FP32 model is evaluated/used
# at, or the INT8 vs FP32 comparison below is meaningless.
THRESHOLD = 0.90


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

model.summary()


# ============================================================
# LOAD REPRESENTATIVE DATA
# ============================================================

print()
print("=" * 60)
print("LOADING REPRESENTATIVE DATA")
print("=" * 60)

data = np.load(TRAIN_FEATURE_PATH)

X_train = data["X"]

print("Original shape:", X_train.shape)

# Add channel dimension
X_train = X_train[..., np.newaxis]

# Same preprocessing used during training
X_train = (X_train + 80.0) / 80.0

X_train = np.clip(
    X_train,
    0.0,
    1.0
)

print("Prepared shape:", X_train.shape)
print(
    "Value range:",
    X_train.min(),
    "->",
    X_train.max()
)


# ============================================================
# REPRESENTATIVE DATASET
# ============================================================
#
# Sample a random, reproducible subset for calibration instead
# of taking the first N rows — the npz array isn't guaranteed
# to be shuffled, so slicing from the front can bias the
# calibration set toward one class and skew the resulting
# activation ranges.

_calib_rng = np.random.default_rng(CALIBRATION_SEED)

_calib_indices = _calib_rng.choice(
    X_train.shape[0],
    size=min(CALIBRATION_SAMPLES, X_train.shape[0]),
    replace=False
)


def representative_dataset():

    for i in _calib_indices:

        sample = X_train[i].astype(
            np.float32
        )

        sample = np.expand_dims(
            sample,
            axis=0
        )

        yield [sample]


# ============================================================
# CONVERTER
# ============================================================

print()
print("=" * 60)
print("CONVERTING TO INT8 TFLITE")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

# Enable optimizations
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# Representative dataset is required
# for full integer quantization.
converter.representative_dataset = (
    representative_dataset
)

# Force integer-only operators
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# Force INT8 input
converter.inference_input_type = tf.int8

# Force INT8 output
converter.inference_output_type = tf.int8


# ============================================================
# CONVERT
# ============================================================

tflite_model = converter.convert()


# ============================================================
# SAVE
# ============================================================

output_path = (
    EXPORT_DIR
    / "vaani_dscnn_v1_int8.tflite"
)

with open(
    output_path,
    "wb"
) as f:

    f.write(
        tflite_model
    )


# ============================================================
# REPORT
# ============================================================

size_bytes = len(tflite_model)
size_kb = size_bytes / 1024

print()
print("=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print()
print("Output:")
print(output_path)

print()
print(
    f"Model size: "
    f"{size_bytes:,} bytes "
    f"({size_kb:.2f} KB)"
)


# ============================================================
# INSPECT TENSORS
# ============================================================

print()
print("=" * 60)
print("TENSOR INFORMATION")
print("=" * 60)

interpreter = tf.lite.Interpreter(
    model_content=tflite_model
)

interpreter.allocate_tensors()

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)


print()
print("INPUT")

for tensor in input_details:

    print(
        "shape:",
        tensor["shape"]
    )

    print(
        "dtype:",
        tensor["dtype"]
    )

    print(
        "quantization:",
        tensor["quantization"]
    )


print()
print("OUTPUT")

for tensor in output_details:

    print(
        "shape:",
        tensor["shape"]
    )

    print(
        "dtype:",
        tensor["dtype"]
    )

    print(
        "quantization:",
        tensor["quantization"]
    )


# ============================================================
# EXPORT QUANTIZATION PARAMETERS
# ============================================================
#
# Deployment code needs these scale/zero_point values to
# quantize incoming features and dequantize the model's
# output back into a probability before applying a threshold.

input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

quant_params = {
    "input": {
        "scale": float(input_scale),
        "zero_point": int(input_zero_point),
        "dtype": "int8"
    },
    "output": {
        "scale": float(output_scale),
        "zero_point": int(output_zero_point),
        "dtype": "int8"
    }
}

quant_params_path = (
    EXPORT_DIR
    / "vaani_dscnn_v1_int8_quant_params.json"
)

with open(
    quant_params_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        quant_params,
        f,
        indent=4
    )

print()
print(f"Saved quantization params: {quant_params_path}")


# ============================================================
# POST-QUANTIZATION EVALUATION
# ============================================================
#
# Confirms accuracy survived quantization by running the
# actual int8 interpreter over the held-out test set, rather
# than only inspecting file size and tensor metadata.

print()
print("=" * 60)
print("POST-QUANTIZATION TEST EVALUATION")
print("=" * 60)

test_data = np.load(TEST_FEATURE_PATH)

X_test = test_data["X"]
y_test = test_data["y"]

X_test = X_test[..., np.newaxis]
X_test = (X_test + 80.0) / 80.0
X_test = np.clip(X_test, 0.0, 1.0)

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

predictions = np.zeros(len(y_test), dtype=np.int64)

for i in range(len(X_test)):

    sample = X_test[i].astype(np.float32)

    # Quantize float input into int8 using the model's
    # own input scale / zero_point.
    sample_q = sample / input_scale + input_zero_point
    sample_q = np.clip(sample_q, -128, 127).astype(np.int8)
    sample_q = np.expand_dims(sample_q, axis=0)

    interpreter.set_tensor(input_index, sample_q)
    interpreter.invoke()

    output_q = interpreter.get_tensor(output_index)[0]

    # Dequantize back to float probabilities.
    probs = (output_q.astype(np.float32) - output_zero_point) * output_scale

    # Apply the same decision threshold used at deployment,
    # not argmax (which is an implicit 0.5 cutoff). probs[1]
    # is P(Vaani) — see model_config.json classes mapping.
    predictions[i] = 1 if probs[1] >= THRESHOLD else 0

accuracy = float(np.mean(predictions == y_test))

confusion = tf.math.confusion_matrix(
    y_test,
    predictions,
    num_classes=2
).numpy()

tn, fp = confusion[0, 0], confusion[0, 1]
fn, tp = confusion[1, 0], confusion[1, 1]

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)

print()
print(f"Decision threshold:  {THRESHOLD}")
print(f"INT8 test accuracy:  {accuracy:.4f}")
print(f"INT8 precision:      {precision:.4f}")
print(f"INT8 recall:         {recall:.4f}")
print(f"INT8 F1:             {f1:.4f}")

print()
print("Confusion matrix (INT8 model):")
print()
print("                 Predicted")
print("                 Negative  Vaani")
print(f"Actual Negative  {tn:8d} {fp:6d}")
print(f"Actual Vaani     {fn:8d} {tp:6d}")

int8_metrics = {
    "test_accuracy": accuracy,
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp)
}

int8_metrics_path = (
    EXPORT_DIR
    / "vaani_dscnn_v1_int8_test_metrics.json"
)

with open(
    int8_metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        int8_metrics,
        f,
        indent=4
    )

print()
print(f"Saved INT8 test metrics: {int8_metrics_path}")


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)