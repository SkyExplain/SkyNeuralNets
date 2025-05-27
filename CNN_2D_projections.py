import healpy as hp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.model_selection import train_test_split

tf.keras.backend.clear_session()  # clear any previous models

data_directory = "/mnt/lustre/scratch/nlsas/home/csic/eoy/ioj/CMBFeatureNet/data/"
os.chdir(data_directory)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF warnings
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

# Read and visualize one map
path_lcdm = "./simulated_maps/lcdm/"
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

path_feature = "./simulated_maps/feature/"
x_raw, y_raw = read_all_maps(path_lcdm, path_feature, n_maps=100)  # 0: lcdm, 1:feature

x_raw_new = np.array(x_raw).squeeze()
imgs = np.array([map_to_image(m) for m in x_raw_new])
if imgs.ndim == 3:
    imgs = imgs[..., np.newaxis]  # add channel dimension

X_train, X_test, y_train, y_test = train_test_split(imgs, y_raw, test_size=0.2, random_state=42)

# ======== Define Model ========
model = models.Sequential([
    layers.Input(shape=imgs.shape[1:]),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary(110)

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

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

plt.figure(figsize=(12, 8))
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.grid()
plt.yscale("log")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"], label="training")
plt.plot(history.history["val_loss"], label="validation")
plt.grid()
plt.yscale("log")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

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
lenghts = [len(true_LCDM), len(true_MoG), len(false_LCDM), len(false_MoG)]

print('     ', 'LCDM', 'FT')
print('True ', len(true_LCDM) / sum(lenghts), len(true_MoG) / sum(lenghts))
print('False', len(false_LCDM) / sum(lenghts), len(false_MoG) / sum(lenghts))
print('--------------')
print("Correct prediction: ", (len(true_LCDM) + len(true_MoG)) / sum(lenghts))
print("Wrong prediction  : ", (len(false_LCDM) + len(false_MoG)) / sum(lenghts))