import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras import layers, models, optimizers, callbacks
import healpy as hp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.model_selection import train_test_split

def read_map(file_path):
    """
    Reads a Healpy map from a FITS file and flattens the data.
    """
    with fits.open(file_path) as hdul:
        hdul.info()
        if len(hdul) > 1 and hasattr(hdul[1], 'columns'):
            print(hdul[1].columns)
        return np.concatenate(hdul[1].data['T'])

# Read and visualize one map
path_lcdm = "./simulated_maps/"
map_temp_data = read_map(path_lcdm + 'cmb_map_0.fits')

nside = hp.npix2nside(len(map_temp_data))
print(f"NSIDE: {nside}")

hp.mollview(map_temp_data, title="Temperature Map", unit="Intensity")
plt.show()

def read_all_maps(path_lcdm, path_feature, n_maps=100):
    maps = []
    labels = []
    # LCDM maps
    for i in range(n_maps):
        map_lcdm = read_map(f"{path_lcdm}cmb_map_{i}.fits")
        maps.append(map_lcdm)
        labels.append(0)  # lcdm
    # Feature maps
    for i in range(n_maps):
        map_feature = read_map(f"{path_feature}cmb_map_feature_{i}.fits")
        maps.append(map_feature)
        labels.append(1)  # feature
    maps = np.array(maps).astype(np.float32)[..., None]  # Add channel dimension
    labels = np.array(labels).astype(np.int32)
    return maps, labels

def map_to_image(hp_map, xsize=256):
    try:
        nside = hp.get_nside(hp_map)
    except Exception as e:
        print("Invalid map shape:", hp_map.shape)
        raise e
    img = hp.mollview(hp_map, xsize=xsize, return_projected_map=True, title="", cbar=False)
    plt.close()
    return img

path_feature = "./simulated_maps/"
x_raw, y_raw = read_all_maps(path_lcdm, path_feature, n_maps=100)  #0: lcdm, 1:feature

x_raw_new = np.array(x_raw).squeeze()
imgs = np.array([map_to_image(m) for m in x_raw_new])
if imgs.ndim == 3:
    imgs = imgs[..., np.newaxis] 

X_train, X_test, y_train, y_test = train_test_split(imgs, y_raw, test_size=0.2, random_state=42)

def create_model(trial, input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
    for i in range(n_conv_layers):
        filters = trial.suggest_categorical(f'filters_{i}', [16, 32, 64, 128])
        kernel_size = trial.suggest_categorical(f'kernel_size_{i}', [3, 5])
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    dense_units = trial.suggest_categorical('dense_units', [64, 128, 256])
    model.add(layers.Dense(dense_units, activation='relu'))

    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid'))

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def objective(trial):
    model = create_model(trial, input_shape=X_train.shape[1:])

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tf_callback = TFKerasPruningCallback(trial, 'val_loss')

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        callbacks=[early_stop, tf_callback],
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_trial.params)
