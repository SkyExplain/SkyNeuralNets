"""
SkyNeuralNets public API.
"""

from .preprocessing import (
    read_map,
    read_all_maps,
    map_to_image,
    compare_maps_spherical,
    compare_maps_patches,
    z_score_norm,
    PCA_norm,
    mollweide_from_cartesian,
)

from .cnn import (
    build_cnn_classifier,
)

from .train import (
    train_model,
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
    "z_score_norm",
    "PCA_norm",
    "mollweide_from_cartesian",
    # model
    "build_cnn_classifier",
    # train
    "train_model",
    # calibration
    "optimal_threshold_from_roc",
    "apply_threshold",
    # confusion
    "confusion_matrix_plot",
    "confusion_matrix_numbers",
]
