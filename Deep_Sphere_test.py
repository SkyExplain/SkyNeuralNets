import tensorflow as tf
import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt
from deepsphere import HealpyGCNN, healpy_layers as hp_layer
from deepsphere import utils
from astropy.io import fits
from sklearn.model_selection import train_test_split

tf.keras.backend.clear_session()

data_directory = "/mnt/lustre/scratch/nlsas/home/csic/eoy/ioj/CMBFeatureNet/data/"
os.chdir(data_directory)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings
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

# Read the data
path_lcdm = "./simulated_maps/lcdm/"
map_temp_data = read_map(path_lcdm + 'cmb_map_0.fits')

# Visualize the map
nside = hp.npix2nside(len(map_temp_data))
print(f"NSIDE: {nside}")

# Plot
hp.mollview(map_temp_data, title="Temperature Map", unit="Intensity")
plt.show()

layers = [
    hp_layer.HealpyChebyshev(K=10, Fout=5, use_bias=True, use_bn=True, activation="relu"),
    hp_layer.HealpyPool(p=1),
    hp_layer.HealpyChebyshev(K=10, Fout=5, use_bias=True, use_bn=True, activation="relu"),
    hp_layer.HealpyPool(p=1),
    hp_layer.HealpyChebyshev(K=10, Fout=5, use_bias=True, use_bn=True, activation="relu"),
    hp_layer.HealpyPool(p=1),
    hp_layer.HealpyChebyshev(K=10, Fout=1),
    tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(tf.reduce_mean(x, axis=1)))
]

indices = np.arange(hp.nside2npix(nside))

model = HealpyGCNN(nside=nside, indices=indices, layers=layers, n_neighbors=20)
batch_size = 16
model.build(input_shape=(None, len(indices), 1))
model.summary(110)

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

path_feature = "./simulated_maps/feature/"
x_raw, y_raw = read_all_maps(path_lcdm, path_feature, n_maps=100)  # 0: lcdm, 1: feature

x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.3, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Memory efficient data loading
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=100).batch(16).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(16)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=1000)

# Function for the probability of classification as LCDM 
def normFeat(p):
    if 0 < p < 0.5:
        rr = 1 - p
    else:
        rr = p
    return rr

# Function for the probability of classification as Feature 
def normLCDM(p):
    if 0 < p < 0.5:
        rr = p
    else:
        rr = 1 - p
    return rr

theory = ['LCDM', 'FT']

true_model = []
pred_model = []
prob_pred_MoG = []
prob_pred_LCDM = []
for i in range(len(x_test)):
    X_test_tf = tf.convert_to_tensor([x_test[i]])
    predictions = model.predict(X_test_tf)
    true_model.append(round(y_test[i]))
    pred_model.append(round(predictions[0][0]))
    prob_pred_MoG.append(round(100 * normFeat(predictions[0][0]), 3))
    prob_pred_LCDM.append(round(100 * normLCDM(predictions[0][0]), 3))

plt.figure(figsize=(12, 8))
plt.plot(history.history["binary_accuracy"], label="training")
plt.plot(history.history["val_binary_accuracy"], label="validation")
plt.grid()
plt.yscale("log")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"], label="training")
plt.plot(history.history["val_loss"], label="validation")
plt.grid()
plt.yscale("log")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Performance: correct & incorrect predictions
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

print('     ', 'LCDM', 'MoG')
print('True ', len(true_LCDM) / sum(lengths), len(true_MoG) / sum(lengths))
print('False', len(false_LCDM) / sum(lengths), len(false_MoG) / sum(lengths))
print('--------------')
print("Correct prediction: ", (len(true_LCDM) + len(true_MoG)) / sum(lengths))
print("Wrong prediction  : ", (len(false_LCDM) + len(false_MoG)) / sum(lengths))