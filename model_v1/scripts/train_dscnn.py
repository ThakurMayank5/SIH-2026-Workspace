"""
VAANI MODEL V1
DS-CNN-Small Training

Input:
    40 x 49 log-Mel spectrogram

Classes:
    0 = Negative
    1 = Vaani

Training data:
    model_v1/features/train.npz

Validation:
    model_v1/features/validation.npz

Test:
    model_v1/features/test.npz
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
CHECKPOINT_DIR = PROJECT_ROOT / "model_v1" / "checkpoints"
EVALUATION_DIR = PROJECT_ROOT / "model_v1" / "evaluation"
EXPORT_DIR = PROJECT_ROOT / "model_v1" / "exports"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# Reproducibility
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

NUM_CLASSES = 2

# Input is:
# 40 Mel bins x 49 time frames
INPUT_SHAPE = (40, 49, 1)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(filename):

    path = FEATURE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    data = np.load(path)

    X = data["X"]
    y = data["y"]

    return X, y


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data():

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    X_train, y_train = load_features("train.npz")
    X_val, y_val = load_features("validation.npz")
    X_test, y_test = load_features("test.npz")

    print()
    print("Raw shapes:")
    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    # --------------------------------------------------------
    # Add channel dimension
    #
    # Before:
    #   (N, 40, 49)
    #
    # After:
    #   (N, 40, 49, 1)
    # --------------------------------------------------------

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # --------------------------------------------------------
    # Normalize log-Mel values
    #
    # Original range:
    #       -80 ... 0 dB
    #
    # Convert to:
    #        0 ... 1
    # --------------------------------------------------------

    X_train = (X_train + 80.0) / 80.0
    X_val = (X_val + 80.0) / 80.0
    X_test = (X_test + 80.0) / 80.0

    X_train = np.clip(X_train, 0.0, 1.0)
    X_val = np.clip(X_val, 0.0, 1.0)
    X_test = np.clip(X_test, 0.0, 1.0)

    print()
    print("Prepared shapes:")
    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    print()
    print("Value range:")
    print(
        f"Train: "
        f"{X_train.min():.3f} -> {X_train.max():.3f}"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# DS-CNN BLOCK
# ============================================================

def ds_cnn_block(
    x,
    filters,
    name
):

    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name=f"{name}_depthwise"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name=f"{name}_dw_bn"
    )(x)

    x = tf.keras.layers.ReLU(
        name=f"{name}_dw_relu"
    )(x)

    # Pointwise convolution
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{name}_pointwise"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name=f"{name}_pw_bn"
    )(x)

    x = tf.keras.layers.ReLU(
        name=f"{name}_pw_relu"
    )(x)

    return x


# ============================================================
# BUILD DS-CNN-SMALL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=INPUT_SHAPE,
        name="log_mel_input"
    )

    # --------------------------------------------------------
    # Initial convolution
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="initial_conv"
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        name="initial_bn"
    )(x)

    x = tf.keras.layers.ReLU(
        name="initial_relu"
    )(x)

    # --------------------------------------------------------
    # DS-CNN blocks
    # --------------------------------------------------------

    x = ds_cnn_block(
        x,
        filters=16,
        name="ds_block_1"
    )

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool_1"
    )(x)

    x = ds_cnn_block(
        x,
        filters=24,
        name="ds_block_2"
    )

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool_2"
    )(x)

    x = ds_cnn_block(
        x,
        filters=32,
        name="ds_block_3"
    )

    x = ds_cnn_block(
        x,
        filters=32,
        name="ds_block_4"
    )

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pool"
    )(x)

    # Small dense layer
    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="dense"
    )(x)

    # Small dropout for regularization
    x = tf.keras.layers.Dropout(
        0.2,
        name="dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classifier"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Vaani_DS_CNN_Small"
    )

    return model


# ============================================================
# MODEL SUMMARY / SIZE
# ============================================================

def calculate_model_size(model):

    trainable = np.sum([
        np.prod(v.shape)
        for v in model.trainable_variables
    ])

    non_trainable = np.sum([
        np.prod(v.shape)
        for v in model.non_trainable_variables
    ])

    total = trainable + non_trainable

    fp32_size = total * 4

    print()
    print("=" * 60)
    print("MODEL SIZE")
    print("=" * 60)

    print(f"Trainable parameters:     {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Total parameters:         {total:,}")

    print()
    print(
        f"Approx FP32 weights: "
        f"{fp32_size / 1024:.2f} KB"
    )

    return int(total)


# ============================================================
# CALLBACKS
# ============================================================

def create_callbacks():

    best_model_path = (
        CHECKPOINT_DIR /
        "best_model.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            str(
                EVALUATION_DIR /
                "training_history.csv"
            )
        )
    ]

    return callbacks


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print()
    print("=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    print()
    print(f"Test loss:     {loss:.6f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion = tf.math.confusion_matrix(
        y_test,
        predictions,
        num_classes=NUM_CLASSES
    ).numpy()

    print()
    print("Confusion matrix:")
    print()
    print("                 Predicted")
    print("                 Negative  Vaani")
    print(
        f"Actual Negative  "
        f"{confusion[0, 0]:8d} "
        f"{confusion[0, 1]:6d}"
    )
    print(
        f"Actual Vaani     "
        f"{confusion[1, 0]:8d} "
        f"{confusion[1, 1]:6d}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    tn = confusion[0, 0]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    tp = confusion[1, 1]

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

    print()
    print("Metrics:")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall / TPR:       {recall:.4f}")
    print(f"F1 score:           {f1:.4f}")
    print(f"False positive rate:{false_positive_rate:.4f}")
    print(f"False negative rate:{false_negative_rate:.4f}")

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "precision": float(precision),
        "recall_tpr": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }

    metrics_path = (
        EVALUATION_DIR /
        "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print()
    print(f"Saved metrics: {metrics_path}")

    return metrics


# ============================================================
# SAVE MODEL CONFIGURATION
# ============================================================

def save_config(parameter_count):

    config = {

        "model": "DS-CNN-Small",

        "keyword": "Vaani",

        "input": {
            "sample_rate": 16000,
            "window_ms": 30,
            "hop_ms": 20,
            "n_fft": 480,
            "n_mels": 40,
            "input_shape": [40, 49, 1]
        },

        "classes": {
            "0": "negative",
            "1": "vaani"
        },

        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED
        },

        "parameters": int(parameter_count)
    }

    path = (
        EVALUATION_DIR /
        "model_config.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=4
        )

    print()
    print(f"Saved config: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("VAANI MODEL V1 — DS-CNN-SMALL")
    print("=" * 60)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = prepare_data()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    model = build_model()

    model.summary()

    parameter_count = calculate_model_size(
        model
    )

    save_config(parameter_count)

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    callbacks = create_callbacks()

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=True,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        EXPORT_DIR /
        "vaani_dscnn_v1.keras"
    )

    model.save(
        final_model_path
    )

    print()
    print(f"Saved final model:")
    print(final_model_path)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("V1 TRAINING COMPLETE")
    print("=" * 60)

    print()
    print("Best checkpoint:")
    print(
        CHECKPOINT_DIR /
        "best_model.keras"
    )

    print()
    print("Final model:")
    print(final_model_path)


if __name__ == "__main__":
    main()