"""
SkyNeuralNets public API.
"""

from .preprocessing import (
    read_map,
    read_all_maps,
    map_to_image,
    compare_maps_spherical,
    compare_maps_patches,
    zscore_norm,
    mollweide_from_cartesian,
    train_val_test_split_strat,
    per_map_standardize,
)

from .architectures import (
    build_cnn_classifier,
    build_mlp_model,
)

from .train import (
    compile_mpl_model,
    train_mpl_model,
    compile_cnn_model,
    train_cnn_model,
)

from .calibration import (
    optimal_threshold_from_roc,
    apply_threshold,
)

from .confusion import (
    confusion_matrix_plot,
    confusion_matrix_numbers,
)

__all__ = [
    # preprocessing
    "read_map",
    "read_all_maps",
    "map_to_image",
    "compare_maps_spherical",
    "compare_maps_patches",
    "zscore_norm",
    "mollweide_from_cartesian",
    "train_val_test_split_strat",
    "per_map_standardize",
    # model
    "build_cnn_classifier",
    "build_mlp_model",
    # train
    "compile_mpl_model",
    "train_mpl_model",
    "compile_cnn_model",
    "train_cnn_model",
    # calibration
    "optimal_threshold_from_roc",
    "apply_threshold",
    # confusion
    "confusion_matrix_plot",
    "confusion_matrix_numbers",
]
