import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def confusion_matrix_numbers(y_true, y_pred):
    """
    Returns the raw confusion matrix with ordering:
    [[TN, FP],
     [FN, TP]]
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return confusion_matrix(y_true, y_pred)

def confusion_matrix_plot(
    y_true,
    y_pred,
    labels=("ΛCDM", "Feature"),
    normalize: bool = True,
    cmap: str = "Blues",
    values_format: str = ".1f",
    figsize=(6, 5),
    fontsize: int = 16,
    savepath: str | None = None,
    dpi: int = 300,
):
    """
    Plot confusion matrix, optionally normalized per true class row.
    """
    cm = confusion_matrix_numbers(y_true, y_pred)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    fig, ax = plt.subplots(figsize=figsize)
    
    #Disable default colorbar
    disp.plot(
        cmap=cmap,
        values_format=values_format,
        ax=ax,
        colorbar=False, 
        im_kw={"vmin": 0.0, "vmax": 1.0} 
    )

    ax.set_xlabel("Predicted label", fontsize=fontsize)
    ax.set_ylabel("True label", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    for text in ax.texts:
        text.set_fontsize(fontsize)

    #Build colorbar manually
    cbar = fig.colorbar(disp.im_, ax=ax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=fontsize)

    pos = cbar.ax.get_position()
    cbar.ax.set_position([
        pos.x0,         
        pos.y0 + 0.01,  
        pos.width,      
        pos.height * 0.97   
    ])
    
    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax