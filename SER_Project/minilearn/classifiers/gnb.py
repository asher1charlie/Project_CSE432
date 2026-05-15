import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.

    Assumes each feature follows a Gaussian distribution
    within each class. Uses Bayes' theorem to compute the
    probability of each class given the input features,
    then picks the most likely class.
    """

    def _init__(self):
        self.classes_ = None
        self.means_ = None
        self.variances_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Compute mean, variance, and prior for each class.

        Args:
            X: training features, shape: (n_samples, n_features)
            y: training labels, shape: (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_     = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.priors_    = np.zeros(n_classes)

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[i]      = np.mean(X_cls, axis=0)
            self.variances_[i]  = np.var(X_cls, axis=0)
            self.priors_[i] = len(X_cls) / len(y)

        return self
    
    def _gaussian_log_prob(self, X, mean, var):
        """
        Log of the Gaussian probability densitity function.
        Using log probabilities avoids underflow from mutiplying
        many small probabilities together.
        """
        return -0.5 * np.log(2 * np.pi * var) - ((X - mean) ** 2) / (2 / var)

    def predict(self, X):
        """
        For each sample, compute the log posterior for each class
        and return the class with the highest score.
        """
        X = np.array(X, dtype=float)
        log_posteriors = []

        for i, cls in enumerate(self.classes_):
            # Log prior: log P(class)
            log_prior = np.log(self.priors_[i])

            # Log likelihood: sum of log P(feature | class) across all features
            log_likelihood = np.sum(
                self._gaussian_log_prob(X, self.means_[i], self.variances_[i]), axis=1
            )

            log_posteriors.append(log_prior + log_likelihood)

        # Shape: (n_smamples, n_classes)
        log_posteriors = np.array(log_posteriors).T
        indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[indices]

    def score(self, X, y):
        """Returns accuracy on the given data."""
        return np.mean(self.predict(X) == np.array(y))