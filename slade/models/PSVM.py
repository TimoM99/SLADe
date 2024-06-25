import numpy as np
from sklearn.svm import SVC

class PSVM:
    def __init__(self, random_state) -> None:
        self.svm = None
        self.random_state = random_state
        np.random.seed(random_state)

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
        labeled_X = X[y != 0] # Filter all labeled data
        redistributed_p = np.array([pr + 0.01 if pr < 0.5 else pr - 0.01 for pr in source_p]) # Weights can not be 0, so shift weights towards 0.5
        if len(labeled_X) != 0:
            X_train = np.tile(labeled_X, (2, 1))
            y_train = np.concatenate((np.full(labeled_X.shape[0], -1), np.ones(labeled_X.shape[0])))
            weights = np.concatenate((redistributed_p[y != 0], 1 - redistributed_p[y != 0]))
            self.svm = SVC(probability=True, random_state=self.random_state)
            self.svm.fit(X_train, y_train, weights)

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
        if self.svm == None:
            return np.full((X.shape[0], 2), 0.5)
        return self.svm.predict_proba(X)

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
        
    
