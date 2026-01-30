import numpy as np
import healpy as hp
from astropy.io import fits
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


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


def map_to_image(hp_map: np.ndarray, xsize: int = 256) -> np.ndarray:
    """
    Converts a Healpy map (1D Npix) to a 2D image using a cartesian projection.

    Returns:
        2D projected image.
    """
    hp_map = np.asarray(hp_map, dtype=np.float64)
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


def z_score_norm(X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray):
    """
    Compute μ, σ on TRAIN ONLY; apply to all splits.

    Returns:
        X_train_scaled, X_test_scaled, X_val_scaled, (mu, std)
    """
    mu = X_train.mean(dtype=np.float64)
    std = X_train.std(dtype=np.float64)
    std = std if std > 0 else 1.0

    def _scale(Z):
        return ((Z - mu) / std).astype(np.float32)

    return _scale(X_train), _scale(X_test), _scale(X_val), (float(mu), float(std))

def PCA_norm(X_train, X_val, X_test, y_train, n_components=100, return_details=False):
    #---- 1. Flatten data ----
    orig_shape = X_train.shape[1:]   # e.g. (H, W, 1) or (H, W)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    #---- 2. Fit PCA on noise (Class 0 in TRAIN only) ----
    noise_indices = (y_train == 0)
    X_noise_flat  = X_train_flat[noise_indices]

    if X_noise_flat.shape[0] == 0:
        raise RuntimeError("No Class 0 maps in y_train, cannot fit PCA")

    print("Fitting PCA on noise maps...")
    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(X_noise_flat)
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    #---- 3. Subtract noise from all datasets ----
    def subtract_noise(X_flat, pca_obj):
        noise_part  = pca_obj.inverse_transform(pca_obj.transform(X_flat))
        signal_part = X_flat - noise_part
        return noise_part, signal_part

    X_train_noise_flat, X_train_signal_flat = subtract_noise(X_train_flat, pca)
    X_val_noise_flat,   X_val_signal_flat   = subtract_noise(X_val_flat,   pca)
    X_test_noise_flat,  X_test_signal_flat  = subtract_noise(X_test_flat,  pca)

    #---- 4. Reshape back (this is what you use for the CNN) ----
    X_train_pca = X_train_signal_flat.reshape(X_train.shape).astype(np.float32)
    X_val_pca   = X_val_signal_flat.reshape(X_val.shape).astype(np.float32)
    X_test_pca  = X_test_signal_flat.reshape(X_test.shape).astype(np.float32)

    print("PCA noise subtraction done.")

    if not return_details:
        return X_train_pca, X_val_pca, X_test_pca

    details = dict(
        orig_shape=orig_shape,
        pca=pca,
        X_train_flat=X_train_flat, X_val_flat=X_val_flat, X_test_flat=X_test_flat,
        X_train_noise_flat=X_train_noise_flat, X_val_noise_flat=X_val_noise_flat, X_test_noise_flat=X_test_noise_flat,
        X_train_signal_flat=X_train_signal_flat, X_val_signal_flat=X_val_signal_flat, X_test_signal_flat=X_test_signal_flat,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    return X_train_pca, X_val_pca, X_test_pca, details

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