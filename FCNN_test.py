import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

tf.keras.backend.clear_session()  # clear any previous models

# Set up environment
data_directory = "/cosmodata/iocampo/SkySimulation/data/"
os.chdir(data_directory)

print("Current working directory:", os.getcwd())

def read_map(file_path):
    """
    Reads a Healpy map from a FITS file and flattens the data.
    """
    with fits.open(file_path) as hdul:
        hdul.info()
        if len(hdul) > 1 and hasattr(hdul[1], 'columns'):
            print(hdul[1].columns)
        return np.concatenate(hdul[1].data['T'])

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

# Read and visualize one map
path_lcdm = "./simulated_maps/"
path_feature = "./simulated_maps/"
map_temp_data = read_map(path_lcdm + 'cmb_map_0.fits')
nside = hp.npix2nside(len(map_temp_data))
print(f"NSIDE: {nside}")
hp.mollview(map_temp_data, title="Temperature Map", unit="Intensity")
plt.show()

# Read all maps
x_raw, y_raw = read_all_maps(path_lcdm, path_feature, n_maps=100)  # 0: lcdm, 1:feature

maps_model = np.atleast_2d(x_raw)
print("Shape of maps_model:", maps_model.shape)

# Normalize
X = (maps_model - np.mean(maps_model, axis=1, keepdims=True)) / np.std(maps_model, axis=1, keepdims=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)

# Define model (Optuna suggestion)
model = models.Sequential()
model.add(layers.Input(shape=(X.shape[1],)))
model.add(layers.Dense(192, activation='relu'))
model.add(layers.Dropout(0.14345855519676617))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.14345855519676617))
model.add(layers.Dense(192, activation='relu'))
model.add(layers.Dropout(0.14345855519676617))
model.add(layers.Dense(320, activation='relu'))
model.add(layers.Dropout(0.14345855519676617))
model.add(layers.Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.SGD(learning_rate=5.598157724953616e-05)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
model.summary(110)

# Train
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Probability functions
def normFeat(p):
    if 0 < p < 0.5:
        rr = 1 - p
    else:
        rr = p
    return rr

def normLCDM(p):
    if 0 < p < 0.5:
        rr = p
    else:
        rr = 1 - p
    return rr

# Predictions
theory = ['LCDM', 'FT']
true_model = []
pred_model = []
prob_pred_MoG = []
prob_pred_LCDM = []
for i in range(len(X_test)):
    X_test_tf = tf.convert_to_tensor([X_test[i]])
    predictions = model.predict(X_test_tf)
    true_model.append(round(y_test[i]))
    pred_model.append(round(predictions[0][0]))
    prob_pred_MoG.append(round(100 * normFeat(predictions[0][0]), 3))
    prob_pred_LCDM.append(round(100 * normLCDM(predictions[0][0]), 3))

# Plot accuracy
plt.figure(figsize=(12, 8))
plt.plot(history.history["binary_accuracy"], label="training")
plt.plot(history.history["val_binary_accuracy"], label="validation")
plt.grid()
plt.yscale("log")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("/cosmodata/iocampo/SkyNeuralNets/plots/model_accuracy_FCNN.png")
plt.show()

# Plot loss
plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"], label="training")
plt.plot(history.history["val_loss"], label="validation")
plt.grid()
plt.yscale("log")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("/cosmodata/iocampo/SkyNeuralNets/plots/model_loss_FCNN.png")
plt.show()

# Performance summary
true_LCDM = []
true_MoG = []
false_LCDM = []
false_MoG = []
for pred, true in zip(pred_model, true_model):
    if pred == 0 and true == 0:
        true_LCDM.append(1)
    if pred == 1 and true == 1:
        true_MoG.append(1)
    if pred == 0 and true == 1:
        false_LCDM.append(1)
    if pred == 1 and true == 0:
        false_MoG.append(1)
lengths = [len(true_LCDM), len(true_MoG), len(false_LCDM), len(false_MoG)]

print('     ', 'LCDM', 'FT')
print('True ', len(true_LCDM) / sum(lengths), len(true_MoG) / sum(lengths))
print('False', len(false_LCDM) / sum(lengths), len(false_MoG) / sum(lengths))
print('--------------')
print("Correct prediction: ", (len(true_LCDM) + len(true_MoG)) / sum(lengths))
print("Wrong prediction  : ", (len(false_LCDM) + len(false_MoG)) / sum(lengths))