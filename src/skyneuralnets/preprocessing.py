import numpy as np
import healpy as hp
from astropy.io import fits
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold


def read_map(file_path: str) -> np.ndarray:
    """
    Reads a Healpy map from a FITS file and flattens the data.

    Args:
        file_path: Path to the FITS file containing the Healpy map.

    Returns:
        Flattened array of the map data.
    """
    with fits.open(file_path) as hdul:
        # If you want the verbose FITS info, uncomment:
        # hdul.info()
        # if len(hdul) > 1 and hasattr(hdul[1], "columns"):
        #     print(hdul[1].columns)
        return np.concatenate(hdul[1].data["T"])


def read_all_maps(
    path_lcdm: str,
    path_feature: str,
    n_maps: int = 100,
    polarization: bool = False,
    file_prefix: str = "cmb_pol_map",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads maps from two directories and returns (maps, labels).

    Expected naming convention (temperature):
        {file_prefix}_T_{i}.fits
        {file_prefix}_T_feature_{i}.fits

    If polarization=True, it expects Q/U too:
        {file_prefix}_Q_{i}.fits, {file_prefix}_U_{i}.fits
        {file_prefix}_Q_feature_{i}.fits, {file_prefix}_U_feature_{i}.fits

    Labels:
        0 -> LCDM
        1 -> feature
    """
    maps = []
    labels = []

    # Temperature maps
    for i in range(n_maps):
        m = read_map(f"{path_lcdm}{file_prefix}_T_{i}.fits")
        maps.append(m)
        labels.append(0)

    for i in range(n_maps):
        m = read_map(f"{path_feature}{file_prefix}_T_feature_{i}.fits")
        maps.append(m)
        labels.append(1)

    labels = np.array(labels).astype(np.int32)

    if not polarization:
        maps = np.array(maps).astype(np.float32)[..., None]  # add channel dim
        return maps, labels

    # Polarization: stack [T, Q, U] as channels
    maps_pol = []
    for i in range(n_maps):
        T_l = read_map(f"{path_lcdm}{file_prefix}_T_{i}.fits")
        Q_l = read_map(f"{path_lcdm}{file_prefix}_Q_{i}.fits")
        U_l = read_map(f"{path_lcdm}{file_prefix}_U_{i}.fits")
        maps_pol.append(np.stack([T_l, Q_l, U_l], axis=-1))

        T_f = read_map(f"{path_feature}{file_prefix}_T_feature_{i}.fits")
        Q_f = read_map(f"{path_feature}{file_prefix}_Q_feature_{i}.fits")
        U_f = read_map(f"{path_feature}{file_prefix}_U_feature_{i}.fits")
        maps_pol.append(np.stack([T_f, Q_f, U_f], axis=-1))

    maps_pol = np.array(maps_pol).astype(np.float32)
    return maps_pol, labels


def map_to_image(hp_map: np.ndarray, xsize: int = 1280) -> np.ndarray:
    """
    Converts a Healpy map (1D Npix) to a 2D image using a cartesian projection.

    Returns:
        2D projected image.
    """
    hp_map = np.asarray(hp_map, dtype=np.float32)
    # validate
    _ = hp.get_nside(hp_map)

    img = hp.cartview(
        hp_map,
        xsize=xsize,
        return_projected_map=True,
        title="",
        cbar=False,
    )
    plt.close()
    return img


def compare_maps_spherical(
    A: np.ndarray,
    B: np.ndarray,
    titleA: str = "Map A",
    titleB: str = "Map B",
    unit: str = "μK",
):
    """
    Mollweide projection for both maps and their difference.
    """
    vmin = np.percentile(np.concatenate([A, B]), 0.5)
    vmax = np.percentile(np.concatenate([A, B]), 99.5)
    diff = B - A

    plt.figure(figsize=(12, 4))
    hp.mollview(A, title=titleA, unit=unit, min=vmin, max=vmax, cmap="coolwarm", sub=(1, 3, 1))
    hp.mollview(B, title=titleB, unit=unit, min=vmin, max=vmax, cmap="coolwarm", sub=(1, 3, 2))
    hp.mollview(diff, title=f"{titleB} - {titleA}", unit=unit, cmap="coolwarm", sub=(1, 3, 3))
    plt.tight_layout()


def compare_maps_patches(
    A: np.ndarray,
    B: np.ndarray,
    titleA: str = "Map A",
    titleB: str = "Map B",
    lon_center: float = 0,
    lat_center: float = 0,
    patch_size: int = 800,
    patch_reso: float = 5,
):
    """
    Gnomonic projections for both maps and their difference.
    """
    vmin = np.percentile(np.concatenate([A, B]), 0.5)
    vmax = np.percentile(np.concatenate([A, B]), 99.5)
    diff = B - A

    plt.figure(figsize=(12, 4))
    hp.gnomview(A, rot=(lon_center, lat_center), xsize=patch_size, reso=patch_reso,
                title=f"{titleA} patch", cmap="coolwarm", min=vmin, max=vmax, sub=(1, 3, 1))
    hp.gnomview(B, rot=(lon_center, lat_center), xsize=patch_size, reso=patch_reso,
                title=f"{titleB} patch", cmap="coolwarm", min=vmin, max=vmax, sub=(1, 3, 2))
    hp.gnomview(diff, rot=(lon_center, lat_center), xsize=patch_size, reso=patch_reso,
                title="Difference patch", cmap="coolwarm", sub=(1, 3, 3))
    plt.tight_layout()

def zscore_per_map(X, eps=1e-6):
    """
    X: (N,H,W) or (N,H,W,C)
    returns same shape
    """
    X = X.astype(np.float32)
    axes = (1, 2)  #normalize per map over spatial pixels
    mean = np.mean(X, axis=axes, keepdims=True)
    std  = np.std(X, axis=axes, keepdims=True)
    return (X - mean) / (std + eps)

def zscore_global_fit(X_train, eps=1e-6):
    X_train = X_train.astype(np.float32)
    mu = np.mean(X_train)
    sig = np.std(X_train)
    return mu, sig + eps

def zscore_global_apply(X, mu, sig):
    X = X.astype(np.float32)
    return (X - mu) / sig

def ensure_channel_dim(X):
    """
    If X is (N,H,W) -> (N,H,W,1)
    If X is (N,H,W,C) -> unchanged
    """
    if X.ndim == 3:
        return X[..., np.newaxis]
    if X.ndim == 4:
        return X
    raise ValueError(f"Unexpected X.ndim={X.ndim}, expected 3 or 4.")

def zscore_norm(X_train, X_val, X_test, mode="per_map"):
    #1) ensure channel dim
    X_train = ensure_channel_dim(X_train)
    X_val   = ensure_channel_dim(X_val)
    X_test  = ensure_channel_dim(X_test)

    #2) normalize
    if mode == "per_map":
        X_train_n = zscore_per_map(X_train)
        X_val_n   = zscore_per_map(X_val)
        X_test_n  = zscore_per_map(X_test)
        norm_info = {"mode": "per_map"}
    elif mode == "global":
        mu, sig = zscore_global_fit(X_train)
        X_train_n = zscore_global_apply(X_train, mu, sig)
        X_val_n   = zscore_global_apply(X_val, mu, sig)
        X_test_n  = zscore_global_apply(X_test, mu, sig)
        norm_info = {"mode": "global", "mu": float(mu), "sig": float(sig)}
    else:
        raise ValueError("mode must be 'per_map' or 'global'")

    return X_train_n, X_val_n, X_test_n, norm_info

def mollweide_from_cartesian(ax, m, title, cmap="RdBu_r", vmin=None, vmax=None):
    # m is (H,W) with lat in [-pi/2, pi/2], lon in [-pi, pi]
    H, W = m.shape
    lon = np.linspace(-np.pi, np.pi, W, endpoint=False)
    lat = np.linspace(-np.pi/2, np.pi/2, H)
    Lon, Lat = np.meshgrid(lon, lat)
    im = ax.pcolormesh(Lon, Lat, m, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=16)
    ax.grid(False)                 #no graticule
    ax.set_xticks([])              #no longitude ticks
    ax.set_yticks([])              #no latitude ticks
    ax.set_xticklabels([])         #no tick labels
    ax.set_yticklabels([])

    ax.set_frame_on(False)         
    return im

def train_val_test_split_strat(X, y, groups, test_size=0.2, val_size=0.2, random_state=12345):
    """
    Splits data into Train, Validation, and Test sets while preserving 
    stratification and ensuring groups are not split across sets.
    
    Args:
        X (np.ndarray): The input features/images.
        y (np.ndarray): The target labels for stratification.
        groups (np.ndarray): The group identifiers to prevent leakage.
        test_size (float): Proportion of total data for the test set.
        val_size (float): Proportion of the *remaining* data for the validation set.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    indices = np.arange(len(y))
    
    #Extract Test Set
    #n_splits is derived from test_size (e.g., 0.2 -> 5 splits)
    test_n_splits = int(1 / test_size)
    sgkf_test = StratifiedGroupKFold(n_splits=test_n_splits, shuffle=True, random_state=random_state)
    trainval_idx, test_idx = next(sgkf_test.split(indices, y, groups=groups))

    #Extract Validation Set from the remainder
    val_n_splits = int(1 / val_size)
    sgkf_val = StratifiedGroupKFold(n_splits=val_n_splits, shuffle=True, random_state=random_state)
    
    #Slice the training/validation pool
    idx_tv = indices[trainval_idx]
    y_tv = y[trainval_idx]
    g_tv = groups[trainval_idx]
    
    train_rel, val_rel = next(sgkf_val.split(idx_tv, y_tv, groups=g_tv))
    
    #Map relative indices back to absolute indices
    train_idx = idx_tv[train_rel]
    val_idx = idx_tv[val_rel]

    #Final Shuffle
    rng = np.random.RandomState(random_state)
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    test_idx = rng.permutation(test_idx)

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx])
    )

def per_map_standardize(X):
    mu = X.mean(axis=(1,2,3), keepdims=True)
    sd = X.std(axis=(1,2,3), keepdims=True) + 1e-8
    return (X - mu) / sd

