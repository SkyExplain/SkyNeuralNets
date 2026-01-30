import numpy as np
from sklearn.metrics import roc_curve, auc

def optimal_threshold_from_roc(y_true, probs):
    """
    Compute ROC curve and choose threshold that maximizes Youden's J statistic: tpr - fpr.

    Returns:
        dict with keys:
            t_opt, auc, fpr, tpr, thresholds
            Apply the new threshold to get predictions, Youden's index https://en.wikipedia.org/wiki/Youden%27s_J_statistic
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    idx = int(np.argmax(j))
    t_opt = float(thr[idx])
    auc_val = float(auc(fpr, tpr))

    return {
        "t_opt": t_opt,
        "auc": auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr,
    }

def apply_threshold(probs, threshold: float):
    """
    Convert probabilities into hard labels using a threshold.
    """
    probs = np.asarray(probs).astype(float)
    return (probs >= threshold).astype(int)
