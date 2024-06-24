import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class GP:
    """
    This class is a wrapper around the Scikit-learn GaussianProcessRegressor class.

    Attributes
    ----------
    random_state : int
        The random seed for this class.
    gp : GaussianProcessRegressor
        The underlying Scikitlearn object.
    fitted : bool

    Methods
    -------
    fit(X, y, source_p)
        Fits the model.
    predict_proba(X)
        Predicts anomaly probability.
    sample(X, n_samples, source_labels, y)
        Samples new labels according to uncertainty sampling.
    """
    def __init__(self, random_state):
        """
        Parameters
        ----------
        random_state : int
            A random seed to initialise this model.
        """
        self.random_state = random_state
        self.gp = None

    def fit(self, X, y, source_p):
        """
        Fit the model.

        Parameters
        ----------
        X : ndarray
            2D array containing input features of the unlabeled data.
        y : ndarray
            1D array containing the labels that we have queried so far.
            Unlabeled data receives the label 0, while anomalous and normal data receive 1 and -1 respectively.
        source_p : ndarray
            1D array containing the human annotator soft labels for all instances.
        """
        labeled_X = X[y != 0]
        if len(labeled_X) != 0:
            # We fit the Gaussian Process only on those data points that have been labeled (i.e., y != 0).
            kernel = Matern(length_scale=0.01, length_scale_bounds=[0.00001, 100000], nu=1/2)
            self.gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=20, random_state=self.random_state, normalize_y=False)
            self.gp.fit(labeled_X, source_p[y != 0])

    def predict_proba(self, X):
        """
        Predict probabilities for X

        Parameters
        ----------
        X : ndarray
            2D array containing the input features of the instances.

        Returns
        -------
        result : ndarray of shape (n_samples, 2)
            2D array containing the anomaly probabilities.
        """
        if self.gp == None:
            return np.full((X.shape[0], 2), 0.5)
        posterior_scores = self.gp.predict(X)
        # Normalise the scores so that probabilities are between 0 and 1.
        if min(posterior_scores) != max(posterior_scores):
            anom_prob = (posterior_scores - min(posterior_scores))/(max(posterior_scores) - min(posterior_scores))
        else:
            anom_prob = posterior_scores
        result = np.array(list(zip(1 - anom_prob, anom_prob)))
        return result

    def sample(self, X, n_samples, source_labels, y):
        """
        Sample new labels for this model according to uncertainty sampling.

        Parameters
        ----------
        X : ndarray
            2D array containing input features of the unlabeled data.
        n_samples : int
            The number of new samples to find.
        source_labels : ndarray
            1D array containing the human annotator hard labels for all instances.
        y : ndarray
            1D array containing the labels that we have queried so far.
            Unlabeled data receives the label 0, while anomalous and normal data receive 1 and -1 respectively.
        """
        assert self.gp != None, 'Fit the model with labels first'
        mean, std = self.gp.predict(X[y == 0], return_std=True)
        scores = np.abs(0.5 - mean)/std
        lowest_scores_index = [x for _, x in sorted(list(zip(scores, np.where(y == 0)[0])), key=lambda x: x[0])][:n_samples]
            
        y[lowest_scores_index] = source_labels[lowest_scores_index]
