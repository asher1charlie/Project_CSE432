import numpy as np

class LogisticRegression:
    """
    Multiclass Logistic Regression using gradient descent.
    Uses One-vs-Rest (OvR) strategy for multiclass classification.

    Args:
        lr: learning rate
        n_iters: number of gradient descent steps
    """

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None # shape: (n_classes, n_features)
        self.bias = None    # shape: (n_classes,)
        self.classes_ = None

    def _sigmoid(self, z):
        """Sigmoid activation: maps any value to (0, 1)."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """
        Train one binary logistic regression per class (OvR).

        Args:
            X: training features, shape (n_samples, n_features)
            Y: training labels, shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)
        
        # Train one binary classifier per class
        for i, cls in enumerate(self.classes_):
            # Treat current class as 1, everything else as 0
            y_binary = (y == cls).astype(float)
            
            w = np.zeros(n_features)
            b = 0.0

            for _ in range(self.n_iters):
                # Forward pass
                z = X @ w + b
                y_hat = self._sigmoid(z)

                # Gradients
                error = y_hat - y_binary
                dw = (X.T @ error) / n_samples
                db = np.mean(error)

                # Update
                w -= self.lr * dw
                b -= self.lr * db

            self.weights[i] = w
            self.bias[i] = b

        return self
    
    def predict_proba(self, X):
        """
        Returns probablity scores for each class
        Shape: (n_samples, n_classes)
        """
        X = np.array(X, dtype=float)
        scores = self._sigmoid(X @ self.weights.T + self.bias)

        return scores
    
    def predict(self, X):
        """Returns the class with the highest probability score."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        
        return self.classes_[indices]

    def score(self, X, y):
        """Returns accuracy on the given data."""
        return np.mean(self.predict(X) == np.array(y))