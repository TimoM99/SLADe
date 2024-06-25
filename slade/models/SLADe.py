from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from pyod.models.iforest import IForest
from scipy.spatial.distance import cdist
import numpy as np
import math

# Setting the number of GP optimizer restarts larger than 0 introduces randomness in different runs of the experiments.

class SLADe:
    """
    The model discussed in the paper.

    Attributes
    ----------
    contamination_factor : float
        The contamination factor of the dataset.
    random_state : int
        The random seed that this model is initialised with.
    q : int
        The hyperparameter q discussed in the paper. It denotes the percentage of instances that should be encapsulated in the averaging Gaussian distribution.
        Read the paper for more details.
    unsupervised_model : pyod.model
        The model calculating unsupervised scores.
    nu : float
        The Matern kernel hyperparameter.

    Methods
    -------
    fit(X, y, source_p)
        Fits the model.
    predict_proba(X)
        Predicts anomaly probability.
    compute_posterior
        Compute posterior scores
    sample(X, n_samples, source_labels, y)
        Samples new labels according to uncertainty sampling.
    """
    def __init__(self, contamination_factor=0.1, random_state=331, q=2, unsupervised_model=None, nu=1/2):
        """
        Initialise SLADe.
        
        Parameters
        ----------
        random_state : int
            The random state of the predictor
        q : int
            The percentage of instances in the averiging hypersphere
        unsupervised_model : pyod.model
            Use an unsupervised model from PyOD or any model with the same interface.
            https://pyod.readthedocs.io/en/latest/pyod.models.html
            Default is IForest.
        nu : float
            The Matern kernel hyperparameter: choose from 1/2, 3/2, 5/2 for optimal algorithm speed.
        """
        np.random.seed(random_state)
        if unsupervised_model == None:
            self.prior_model = IForest(contamination=contamination_factor, random_state=random_state)
        else:
            self.prior_model = unsupervised_model
        self.nu = nu
        self.q = q
        self.random_state = random_state
        self.gp = None
        
        

    def fit(self, X, y, source_p):
        """
        Fit SLADe.
        
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
        self.X = X
        self.y = y
        self.prior_model.fit(X)
        prior = self.prior_model.predict_proba(X)[:, 1] # Uses the built in normalisation scaler by default.
        
        labeled_X = X[y != 0]
        if len(labeled_X) > 0:
            train_scores = source_p[y != 0] - prior[y != 0] # Calculate the deviation for the acquired soft labels so far.
            kernel = Matern(length_scale=0.01, length_scale_bounds=[0.00001, 10000], nu=self.nu)
            # Setting the number of optimizer restarts larger than 0 introduces randomness in different runs of the experiment.
            self.gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=20, random_state=self.random_state, normalize_y=False)
            self.gp.fit(labeled_X, train_scores)

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
            return self.prior_model.predict_proba(X)
        posterior_disributions = self.compute_posterior(X)
        posterior_scores = np.array([x[0] for x in posterior_disributions])
        # Scores are min-max normalised so that they are in the range 0 and 1, other scalers could be used such as unify, squash, etc.
        if min(posterior_scores) != max(posterior_scores):
            anom_prob = (posterior_scores - min(posterior_scores))/(max(posterior_scores) - min(posterior_scores))
        else:
            anom_prob = [posterior_scores]
        result = np.array(list(zip(1 - anom_prob, anom_prob)))
        return result


    def compute_posterior(self, X):
        """
        Compute posterior scores for instances.

        Parameters
        ----------
        X : ndarray
            2D array containing the input features of the instances.
    
        Returns
        -------
        result : ndarray of shape (n_samples, 2)
            2D array containing the mean prediction and standard deviation for every instance. 
        """
        prior = self.prior_model.predict_proba(X)[:, 1]
        dist = cdist(X.astype('float64'), self.X.astype('float64'), metric='euclidean')

        result = np.full((len(X), 2), None)
        for i, x in enumerate(X):

            # If model is predicted anomalous by unsupervised model, don't average out deviation
            if self.prior_model.predict(x.reshape(1, -1))[0] == 1:
                result[i] = prior[i] + self.gp.predict(x.reshape(1, -1))[0]
                continue

            # Define standard deviation of averaging Gaussian distribution
            REQUESTED_PCT = self.q/100
            NB_DATA_NEEDED = int(REQUESTED_PCT * len(self.X))
            sorted_dist = sorted(dist[i])
            BALL_STD = sorted_dist[NB_DATA_NEEDED]/3

            # Sample over the Gaussian surface
            NB_SAMPLES = 100
            samples = np.random.multivariate_normal(x.astype(np.float64), np.diag(np.full(len(x), BALL_STD**2)), NB_SAMPLES)

            # Calculate mean function value and cov matrix for samples
            samples_mean, samples_cov = self.gp.predict(samples, return_cov=True)

            new_mean = prior[i] + 1/NB_SAMPLES * np.sum(samples_mean)
            new_variance = 1/(NB_SAMPLES**2) * np.sum(samples_cov)
            result[i] = np.array([new_mean, math.sqrt(new_variance)])
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
        prior = self.prior_model.predict_proba(X[y == 0])[:, 1]
        scores = np.abs(0.5 - mean - prior)/std
        lowest_scores_index = [x for _, x in sorted(list(zip(scores, np.where(y == 0)[0])), key=lambda x: x[0])][:n_samples]
            
        y[lowest_scores_index] = source_labels[lowest_scores_index]

