# SLADe
This is a Github repository containing code for SLADe [1]. Additionally, it contains code for the other methods used to compare SLADe against, two scripts to run experiments, two visualisation scripts and 21 [datasets](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/) [2].

## Abstract
Anomaly detection aims at detecting examples that do not conform to normal behavior. Increasingly, anomaly detection is being approached from a semi-supervised perspective where active learning is employed to acquire a small number of strategically selected labels. However, because anomalies are not always well-understood events, the user may be uncertain about how to label certain instances. Thus, one can relax this request and allow the user to provide soft labels (i.e., probabilistic labels) that represent their belief that a queried example is anomalous. These labels are naturally noisy due to the user's inherent uncertainty in the label and the fact that people are known to be bad at providing well-calibrated probability instances. To cope with these challenges, we propose to exploit a Gaussian Process to learn from actively acquired soft labels in the context of anomaly detection. This enables leveraging information about nearby examples to smooth out possible noise. Empirically, we compare our proposed approach to several baselines on 21 datasets and show that it outperforms them in the majority of experiments. 

## Contents and usage
To use SLADe or any other model compared against in the paper, you can simply use pip to install.
```bash
pip install git+https://github.com/TimoM99/SLADe.git
```

Example usage looks like this
```python
from pyod.models.iforest import IForest
from slade.models import SLADe

# Instantiate a PyOD unsupervised anomaly detection model
# Or any other unsupervised model using the same interface.
iforest = IForest(contamination_factor=gamma)
# Instantiate the SLADe model
model = SLADe(q=2, unsupervised_model=iforest, nu=1/2)
# Fit the model: 
# - p are the soft labels
# - y is ignored, but used for consistency over models (SSDO & HIF)
model.fit(X, y, p)
# Predict using the model
y_pred = model.predict_proba(X_test)
```

## Reproducing experimental results
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

## Dependencies
SLADe requires the following packages:
- [Scikit-learn](https://scikit-learn.org/stable/)
- [PyOD](https://pyod.readthedocs.io/en/latest/index.html)
- [SciPy](https://scipy.org/)
- [NumPy](https://numpy.org/doc/stable/index.html)

## Contact
Feedback for improvement of the code or documentation is appreciated. My inbox is also open for any questions!

Contact: timo.martens@kuleuven.be


## References
[1]: Martens, T., Perini, L., & Davis, J. (2023). *Semi-supervised learning from active noisy soft labels for anomaly detection*. In Proceedings of European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases. Springer.

[2]: Campos, G. O., Zimek A., Sander J., Campello R. J. G. B., Micenková B., Schubert E., Assent I. and Houle M. E.  (2016). *On the Evaluation of Unsupervised Outlier Detection: Measures, Datasets, and an Empirical Study*. Data Mining and Knowledge Discovery 30(4): 891-927, DOI: 10.1007/s10618-015-0444-8
