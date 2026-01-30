# src/skyneurals/cnn.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_cnn_classifier(
    input_shape: Tuple[int, int, int],
    filters: Sequence[int] = (32, 64, 128),
    dense_units: int = 64,
    dropout: float = 0.2,
    l2: float = 1e-4,
    learning_rate: float = 1e-4,
    clipnorm: Optional[float] = 1.0,
    metrics: Optional[list] = None,
):
    """
    Build + compile the *same CNN* as in the original notebook:

      Input
      [Conv(32) + BN + ReLU + MaxPool]
      [Conv(64) + BN + ReLU + MaxPool]
      [Conv(128) + BN + ReLU]
      GlobalAvgPool
      Dropout(dropout)
      Dense(dense_units, relu, l2)
      Dropout(dropout)
      Dense(1, sigmoid)

    Parameters you usually tune:
      - filters: (32,64,128) etc.
      - dropout: 0.0–0.6
      - l2: 0–1e-3
      - learning_rate: 1e-5–3e-4
      - dense_units: 32, 64, 128 ...

    Returns:
      Compiled tf.keras.Model
    """

    if metrics is None:
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

    f1, f2, f3 = filters

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(f1, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(f2, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(f3, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout),

        layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2) if l2 and l2 > 0 else None,
        ),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=metrics)
    return model
