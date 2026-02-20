from __future__ import annotations
from tabnanny import verbose
from typing import Optional
import tensorflow as tf

def compile_mpl_model(model, learning_rate=1e-3):
    """
    Configures the model's solver and metrics.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.AUC(name="auc", from_logits=True)]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def train_mpl_model(model, X_train, y_train, validation_data, 
                 epochs=2000, batch_size=32, patience=100, verbose=1):
    """
    Executes the training loop with EarlyStopping.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="auc",
            patience=patience,
            restore_best_weights=True,
            mode="max"
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history

def compile_cnn_model(model, learning_rate: float = 1e-3, metrics: list = None):
    """
    Configures the CNN with an optimizer, loss, and metrics.
    """
    if metrics is None:
        metrics = [tf.keras.metrics.AUC(name="auc", from_logits=True)]
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Using from_logits=True as it's generally more numerically stable
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def train_cnn_model(
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
    Handles the fit process with EarlyStopping and ReduceLROnPlateau.
    """
    # Logic to determine if we want to maximize (AUC/Acc) or minimize (Loss)
    mode = "max" if any(x in monitor.lower() for x in ["auc", "acc", "accuracy"]) else "min"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            mode=mode,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor_rlrop,
            patience=patience_rlrop,
            mode=mode,
            min_lr=min_lr,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )
    return history