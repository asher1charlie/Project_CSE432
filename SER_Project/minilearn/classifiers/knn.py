import numpy as np

class KNN:
    """
    K-Nearest Neighbors classifier.

    For each test sample, finds the k closest training samples using
    Euclidean distance and returns the most common class among them.

    Args:
        k: number of neighbors to consider (default 5)
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Args:
            X: training features, shape (n_samples, n_features)
            y: training labels, shape (n_samples,)
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        return self

    def _euclidean_distnace(self, a, b):
        """Euclidean distance between two vectors."""
        return np.sqrt(np.sum(a - b) **2)
    
    def predict(self, X):
        """
        For each test sample, find the k nearest training samples
        and return the most common label among them.

        Args:
            X: test features, shape (n_samples, n_features)

        Returns:
            predicted labels, shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        predictions = []

        for sample in X:
            # Compute distance from this sample to every training sample
            distances = [self._euclidean_distnace(sample, x) for x in self. X_train]
            distances = np.array(distances)

            # Get indices of the k smallest distances
            k_indices = np.argsort(distances)[:self.k]

            # Get the labels of those k neighbors 
            k_labels = self.y_train[k_indices]

            # Return the most common label
            labels, counts = np.unique(k_labels, return_counts=True)
            predictions.append(labels[np.argmax(counts)])

        return np.array(predictions)

    def score(self, X, y):
        """Returns accuracy on the given data"""
        return np.mean(self.predict(X) == np.array(y))