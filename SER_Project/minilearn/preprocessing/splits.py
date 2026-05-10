import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits X and y into random train and test subset.

    Args:
        X: feature array, shape (n_samples, n_features)
        y: label array, shape (n_samples,)
        test_size: fraction of data to use testing (default 0.2)
        random_state: seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
    """
    X = np.array(X)
    y = np.array(y)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]