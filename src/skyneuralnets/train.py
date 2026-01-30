from __future__ import annotations
from typing import Optional

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 1000,
    batch_size: int = 32,
    monitor: str = "val_auc",
    patience_es: int = 300,
    patience_rlrop: int = 300,
    factor_rlrop: float = 0.5,
    min_lr: float = 1e-6,
    verbose: int = 1,
):
    """
    Train a compiled Keras model with sensible defaults and your callback setup.
    """
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            mode="max" if "auc" in monitor or "acc" in monitor else "min",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor_rlrop,
            patience=patience_rlrop,
            mode="max" if "auc" in monitor or "acc" in monitor else "min",
            min_lr=min_lr,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose,
    )
    return history
