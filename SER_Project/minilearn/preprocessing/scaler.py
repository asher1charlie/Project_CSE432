import numpy as np

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    z = (x - mean) / std
    """

    def __init__(self):
        self.mean = None
        self.std_ = None

    def fit(self, X):
        """Compute mean and std from training data."""
        X = np.array(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        self.std_[self.std_ == 0] = 1
        return self
    
    def transform(self, X):
        """Apply standardization using the fitted mean and std."""
        X = np.array(X, dtype=float)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)