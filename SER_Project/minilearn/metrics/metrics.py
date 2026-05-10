import numpy as np

def accuracy_score(y_true, y_pred):
    """Fraction of correctly classified samples."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix (y_true, y_pred):
    """
    Returns a matrix where entry [i, j] is the number of samples with
    true label i predicted as label j.
    """

    