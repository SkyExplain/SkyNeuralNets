import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt
from deepsphere import HealpyGCNN, healpy_layers as hp_layer
from deepsphere import utils
from healpy import read_map

data_directory = "/mnt/lustre/scratch/nlsas/home/csic/eoy/ioj/CMBFeatureNet/data/"
os.chdir(data_directory)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #suppress TF warnings
print("Current working directory:", os.getcwd())

nmaps = 100
from astropy.io import fits
def read_map(file_path):
    """
    Reads a Healpy map from a FITS file and flattens the data.
    """
    
    with fits.open(file_path) as hdul:
        hdul.info()
        if len(hdul) > 1 and hasattr(hdul[1], 'columns'):
            print(hdul[1].columns)
        return np.concatenate(hdul[1].data['T'])

def read_all_maps(path_lcdm, path_feature, n_maps=nmaps):
    maps = []
    labels = []
    
    #LCDM maps
    for i in range(n_maps):
        map_lcdm = read_map(f"{path_lcdm}cmb_map_{i}.fits")
        maps.append(map_lcdm)
        labels.append(0)  #lcdm
    
    #Feature maps
    for i in range(n_maps):
        map_feature = read_map(f"{path_feature}cmb_map_feature_{i}.fits")
        maps.append(map_feature)
        labels.append(1)  #feature
    
    maps = np.array(maps).astype(np.float32)[..., None]  #Add channel dimension
    labels = np.array(labels).astype(np.int32)
    #print(labels)
    return maps, labels

#Read the data
path_lcdm = "./simulated_maps/lcdm/"
map_temp_data = read_map(path_lcdm + 'cmb_map_0.fits')
nside = nside = hp.npix2nside(len(map_temp_data))
indices = np.arange(hp.nside2npix(nside))
path_feature = "./simulated_maps/feature/"
x_raw, y_raw = read_all_maps(path_lcdm, path_feature, n_maps=nmaps) #0: lcdm, 1:feature

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.3, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=100).batch(16).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(16)

def objective(trial):
    tf.keras.backend.clear_session()

    # Hyperparameters to search
    num_blocks = trial.suggest_int("num_blocks", 2, 5)
    K = trial.suggest_categorical("K", [3, 5, 7, 10])
    base_Fout = trial.suggest_categorical("base_Fout", [4, 8, 16])
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu"])
    use_bn = trial.suggest_categorical("use_bn", [True, False])
    use_dropout = trial.suggest_categorical("use_dropout", [True, False])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    # Build the model layers
    layers = []
    Fout = base_Fout
    for i in range(num_blocks):
        layers.append(
            hp_layer.HealpyChebyshev(K=K, Fout=Fout, use_bias=True, use_bn=use_bn, activation=activation)
        )
        if use_dropout:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
        layers.append(hp_layer.HealpyPool(p=1))
        Fout = max(2, Fout // 2)

    # Final conv and classification
    layers.append(hp_layer.HealpyChebyshev(K=K, Fout=1))
    layers.append(tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(tf.reduce_mean(x, axis=1))))

    # Build the model
    model = HealpyGCNN(nside=nside, indices=indices, layers=layers, n_neighbors=20)
    model.build(input_shape=(None, len(indices), 1))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    # Train
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    print("model training finished")
    val_acc = max(history.history['val_binary_accuracy'])
    return val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print(f"  Params: {trial.params}")

#Save top 10 trials
#top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:10]

#with open("optuna_top10_trials.txt", "w") as f:
#    for i, t in enumerate(top_trials):
#        f.write(f"Trial #{t.number}\n")
#        f.write(f"  Accuracy: {t.value:.4f}\n")
#        f.write("  Params:\n")
#        for key, value in t.params.items():
#            f.write(f"    {key}: {value}\n")
#        f.write("\n")