import numpy as np
import anomatools
from anomatools.models.scalers import SquashScaler
from pyod.models.iforest import IForest

class SSDO:
    """
    Wrapper around the SSDO model in https://github.com/Vincent-Vercruyssen/anomatools

    Attributes
    ----------
    ssdo : anomatools.model
        The SSDO model
    random_state : int
        The random seed that this model is initialised with.
    contamination_factor : float
        The contamination factor of the dataset

    Methods
    -------
    fit(X, y, source_p)
        Fits the model.
    predict_proba(X)
        Predicts anomaly probability.
    sample(X, n_samples, source_labels, y)
        Samples new labels according to uncertainty sampling.
    """

    def __init__(self, contamination_factor, random_state):
        """
        Initialise SSDO.

        Parameters
        ----------
        contamination_factor : float
            The contamination factor of the dataset.
        random_state : int
            The random seed to initialise this model with.
        """
        self.ssdo = None
        self.random_state = random_state
        self.contamination_factor = contamination_factor
        np.random.seed(random_state)

    # parameter p is not used, but used for consistency with other models.
    def fit(self, X, y, source_p):
        """
        Fit SSDO.
        
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
        self.ssdo = anomatools.models.SSDO(
            base_detector=IForest(contamination=self.contamination_factor, random_state=self.random_state), 
            scaler=SquashScaler(self.contamination_factor))
        self.ssdo.fit(X, y)

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
        return self.ssdo.predict_proba(X)

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
        unlabeled_index = np.where(y == 0)[0]
        np.random.shuffle(unlabeled_index)
        pred_prob = self.predict_proba(X[unlabeled_index])[:, 1]
        most_uncertain_index = [x for _, x in sorted(list(zip(pred_prob, unlabeled_index)), key=lambda x: abs(x[0] - 0.5))][:n_samples]
        y[most_uncertain_index] = source_labels[most_uncertain_index]

