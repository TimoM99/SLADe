# SLADe
This is a Github repository containing code for SLADe [1]. Additionally, it contains code for the other methods used to compare SLADe against, two scripts to run experiments, two visualisation scripts and 21 datasets [2].

## Abstract
Anomaly detection aims at detecting examples that do not conform to normal behavior. Increasingly, anomaly detection is being approached from a semi-supervised perspective where active learning is employed to acquire a small number of strategically selected labels. However, because anomalies are not always well-understood events, the user may be uncertain about how to label certain instances. Thus, one can relax this request and allow the user to provide soft labels (i.e., probabilistic labels) that represent their belief that a queried example is anomalous. These labels are naturally noisy due to the user's inherent uncertainty in the label and the fact that people are known to be bad at providing well-calibrated probability instances. To cope with these challenges, we propose to exploit a Gaussian Process to learn from actively acquired soft labels in the context of anomaly detection. This enables leveraging information about nearby examples to smooth out possible noise. Empirically, we compare our proposed approach to several baselines on 21 datasets and show that it outperforms them in the majority of experiments. 

## Contents and usage
This repository contains:
- The models folder, with, amongst others, ``SLADe.py`` containing the SLADe class.
- Two experiment_*.py scripts that run the experiments from the paper.
- Two plot_*.py scripts that visualize the results from the experiments.
- The datasets folder containing all datasets used in the experiments, or you can find them [here](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/).

## SLADe.py
SLADe uses a dataset (X, p) of size $n$. For reasons of consistency between the different models, our implementation requires three inputs to fit SLADe.
- X: An array of size $n\times d$ containing the input features of the dataset (both labeled and unlabeled).
- y: An array of size $n$ that shows which datapoints are labeled (1 is positive, 0 is unlabeled and -1 is negative). SLADe is indifferent between positives and negatives.
- p: An array of size $n$ containing the soft labels for the labeled datapoints and any value otherwise.
Given these three input features and the contamination factor ``gamma`` of the data, we can fit and predict our model.
```python
from pyod.models.iforest import IForest

# Instantiate a PyOD unsupervised anomaly detection model
iforest = IForest(contamination_factor=gamma)
# Instantiate the SLADe model
model = SLADe(q=2, unsupervised_model=iforest, nu=1/2)
# Fit the model
model.fit(X, y, p)
# Predict using the model
y_pred = model.predict_proba(X_test)
```

## Dependencies
SLADe requires the following packages:
- [Scikit-learn](https://scikit-learn.org/stable/)
- [PyOD](https://pyod.readthedocs.io/en/latest/index.html)
- [SciPy](https://scipy.org/)
- [NumPy](https://numpy.org/doc/stable/index.html)

## Running experiments
This repository contains the ``experiment_varying_noise.py`` and ``experiment_varying_hyperparameter`` empirical evaluation scripts.
To run these, one should provide the correct arguments in their terminal.
```console
python3 experiment_varying_noise.py -dataset -n_random_states -n_samples -label_budget -sample_size
```
- `dataset` = The path to the dataset in the datasets folder
- `n_random_states` = The number of times the random seed should be varied, 5 in the paper.
- `n_samples` = The number of times we should sample hard labels per random seed, 20 in the paper.
- `label_budget` = The ratio of all the data that should end up labeled, 0.6 in the paper.
- `sample_size` = The ratio of instances to label at once, 0.05 in the paper.
```console
python3 experiment_varying_hyperparameter.py -dataset -n_random_states -n_samples -label_budget -sample_size -hyperparameter
```
- `hyperparameter` = The hyperparameter of SLADe that should vary (i.e., `q`, `nu` or `prior`).

## Contact
Feedback for improvement of the code or documentation is appreciated. My inbox is also open for any questions!

Contact the paper author: timo.martens@kuleuven.be


## References
[1]: Martens, T., Perini, L., & Davis, J. (2023). *Semi-supervised learning from active noisy soft labels for anomaly detection*. In Proceedings of European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases. Springer.

[2]: Campos, G. O., Zimek A., Sander J., Campello R. J. G. B., Micenková B., Schubert E., Assent I. and Houle M. E.  (2016). *On the Evaluation of Unsupervised Outlier Detection: Measures, Datasets, and an Empirical Study*. Data Mining and Knowledge Discovery 30(4): 891-927, DOI: 10.1007/s10618-015-0444-8
