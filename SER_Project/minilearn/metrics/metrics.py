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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    n = len(classes)

    # Map class labels to indices 0..n-1
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[label_to_idx[true]][label_to_idx[pred]] += 1
    
    return matrix

def precision_score(y_true, y_pred, average="macro"):
    """
    Precision = TP / (TP + FP) per class, then averaged.
    average: 'macro' (unweighted mean) or 'weighted' (weighted by support)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)

    precisions = []
    supports = []

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        supports.append(np.sum(y_true == c))

    precisions = np.array(precisions)
    supports   = np.array(supports)

    if average == "weighted":
        return np.sum(precisions * supports) / np.sum(supports)
    return np.mean(precisions) # macro

def recall_score(y_true, y_pred, average="macro"):
    """
    Recall = TP / (TP + FN) per class, then averaged.
    average: 'macro' (unweighted mean) or 'weighted' (weighted by support)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)

    recalls = []
    supports = []

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        supports.append(np.sum(y_true == c))

    recalls  = np.array(recalls)
    supports = np.array(supports)

    if average == "weighted":
        return np.sum(recalls * supports) / np.sum(supports)
    return np.mean(recalls) # macro

def f1_score(y_true, y_pred, average="macro"):
    """
    F1 = 2 * (precision * recall) / (precision + recall) per class, then averaged.
    average: 'macro' (unweighted mean) or 'weighted' (weighted by support)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)

    f1s = []
    supports = []

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1s.append(f1)
        supports.append(np.sum(y_true == c))

    f1s = np.array(f1s)
    supports = np.array(supports)

    if average == "weighted":
        return np.sum(f1s * supports) / np.sum(supports)
    return np.mean(f1s) # macro
    