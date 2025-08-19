from scipy.stats import skewnorm
import numpy as np
import healpy as hp
import csv
import os
from astropy.io import fits
from matplotlib import pyplot as plt

def read_map(file_path):
    """
    Reads a Healpy map from a FITS file and flattens the data.
    Args:
        file_path (str): Path to the FITS file containing the Healpy map.
    Returns:
        np.ndarray: Flattened array of the map data.
    """

    with fits.open(file_path) as hdul:
        hdul.info()
        if len(hdul) > 1 and hasattr(hdul[1], 'columns'):
            print(hdul[1].columns)
        return np.concatenate(hdul[1].data['T'])
    
def read_all_maps(path_lcdm, path_feature, n_maps=100):
    """Reads all maps from specified directories and returns them as numpy arrays.
    Args:
        path_lcdm (str): Path to the directory containing LCDM maps.
        path_feature (str): Path to the directory containing feature maps.
        n_maps (int): Number of maps to read from each directory.
    Returns:
        tuple: A tuple containing two numpy arrays - maps and labels.
    """

    maps = []
    labels = []
    
    #LCDM maps
    for i in range(n_maps):
        map_lcdm = read_map(f"{path_lcdm}cmb_map_{i}.fits")
        maps.append(map_lcdm)
        labels.append(0)  #lcdm labeled 0
    
    #Feature maps
    for i in range(n_maps):
        map_feature = read_map(f"{path_feature}cmb_map_feature_{i}.fits")
        maps.append(map_feature)
        labels.append(1)  #feature labeled 1
    
    maps = np.array(maps).astype(np.float32)[..., None]  #channel dimension
    labels = np.array(labels).astype(np.int32)
    return maps, labels

def map_to_image(hp_map, xsize=256):
    """ Converts a Healpy map to a 2D image.
    Args:
        hp_map (np.ndarray): The Healpy map to convert.
        xsize (int): The size of the output image.
    Returns:
        np.ndarray: The 2D image representation of the Healpy map.
    Raises:
        Exception: If the map shape is invalid.
    """

    #Validate that the map has correct length
    hp_map = np.asarray(hp_map, dtype=np.float64)
    try:
        nside = hp.get_nside(hp_map)
    except Exception as e:
        print("Invalid map shape:", hp_map.shape)
        raise e
    img = hp.cartview(hp_map, xsize=xsize, return_projected_map=True, title="", cbar=False)
    plt.close()
    return img

def compare_maps_spherical(A, B, titleA="Map A", titleB="Map B", unit="μK", 
                            lon_center=0, lat_center=0, patch_size=800, patch_reso=5):
    """
    Compare two HEALPix maps on the sphere:
    - Mollweide projection for both maps and their difference
    Args:
        A (np.ndarray): First map to compare.
        B (np.ndarray): Second map to compare.
        titleA (str): Title for the first map.
        titleB (str): Title for the second map.
        unit (str): Unit of the map values.
        lon_center (float): Longitude center for the Mollweide projection.
        lat_center (float): Latitude center for the Mollweide projection.
        patch_size (int): Size of the patch for the Mollweide projection.
        patch_reso (int): Resolution of the Mollweide projection.
    Returns:
        None: Displays the comparison maps.
    """
    #common scale for fair visual comparison
    vmin = np.percentile(np.concatenate([A, B]), 0.5)
    vmax = np.percentile(np.concatenate([A, B]), 99.5)
    diff = B - A

    plt.figure(figsize=(12,4))
    hp.mollview(A, title=titleA, unit=unit, min=vmin, max=vmax, cmap="coolwarm", sub=(1,3,1))
    hp.mollview(B, title=titleB, unit=unit, min=vmin, max=vmax, cmap="coolwarm", sub=(1,3,2))
    hp.mollview(diff, title=f"{titleB} - {titleA}", unit=unit, cmap="coolwarm", sub=(1,3,3))
    plt.tight_layout()

def compare_maps_patches(A, B, titleA="Map A", titleB="Map B", 
                      lon_center=0, lat_center=0, patch_size=800, patch_reso=5):
    """ Compare two HEALPix maps in zoomed patches:
    - Gnomonic projections for both maps and their difference
    Args:
        A (np.ndarray): First map to compare.
        B (np.ndarray): Second map to compare.
        titleA (str): Title for the first map.
        titleB (str): Title for the second map.
        lon_center (float): Longitude center for the Gnomonic projection.
        lat_center (float): Latitude center for the Gnomonic projection.
        patch_size (int): Size of the patch for the Gnomonic projection.
        patch_reso (int): Resolution of the Gnomonic projection.
    Returns:
        None: Displays the comparison patches.
    """
    #common scale for fair visual comparison
    vmin = np.percentile(np.concatenate([A, B]), 0.5)
    vmax = np.percentile(np.concatenate([A, B]), 99.5)
    diff = B - A

    plt.figure(figsize=(12,4))
    hp.gnomview(A, rot=(lon_center, lat_center), xsize=patch_size, reso=patch_reso, 
                title=f"{titleA} patch", cmap="coolwarm", min=vmin, max=vmax, sub=(1,3,1))
    hp.gnomview(B, rot=(lon_center, lat_center), xsize=patch_size, reso=patch_reso, 
                title=f"{titleB} patch", cmap="coolwarm", min=vmin, max=vmax, sub=(1,3,2))
    hp.gnomview(diff, rot=(lon_center, lat_center), xsize=patch_size, reso=patch_reso, 
                title="Difference patch", cmap="coolwarm", sub=(1,3,3))
    plt.tight_layout()

def z_score_norm(X_train, X_test, X_val):

    """Compute μ,σ on TRAIN ONLY; apply to all splits. 
    Args:
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Test data.
        X_val (np.ndarray): Validation data.
    Returns:
        tuple: Scaled arrays for training, test, and validation data, along with (μ, σ).
    Raises:
        None: Returns scaled arrays and the mean and standard deviation.
    """
    mu  = X_train.mean(dtype=np.float64)
    std = X_train.std(dtype=np.float64)
    # avoid divide-by-zero
    std = std if std > 0 else 1.0

    def _scale(Z): return ((Z - mu) / std).astype(np.float32)
    return _scale(X_train), _scale(X_test),  _scale(X_val), (float(mu), float(std))