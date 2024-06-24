from turtle import color
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# Non-ideal sampling of training data right now,
# because sampled posterior scores are used to calculate prediction probability.
def distance(points, initial=0.1):
    nb_p = len(points)
    results = np.zeros((nb_p * nb_p, points.shape[1]))
    for i in range(nb_p):
        for j in range(nb_p):
            if i == j:
                continue
            results[i * nb_p + j] = abs(points[i] - points[j])

    return np.amin(results, axis=0, where=results != 0, initial=initial), np.amax(results, axis=0, where=results != 0, initial=initial)

def max_distance(points):
    dist = cdist(points, points, metric='euclidean')
    return np.max(dist)

def min_distance(points):
    dist = cdist(points, points, metric='euclidean')

    return np.min(dist[dist != 0])

def probabilistic_auc(label_distributions, p_distributions, n_auc_samples):
    auc_scores = np.zeros(n_auc_samples)
    for k in range(n_auc_samples):
        prob = [np.random.normal(x[0], x[1]) for x in p_distributions]
        limiter = lambda x : min(1, max(x, 0))
        anom_prob = np.array(list(map(limiter, prob)))

        samples = np.random.rand(len(label_distributions))
        sampled_y = [1 if p > samples[i] else -1 for i, p in enumerate(label_distributions)]
        
        score = roc_auc_score(sampled_y, anom_prob)
        auc_scores[k] = score
    avg_score = np.average(auc_scores)
    std_score = np.std(auc_scores)
    
    return avg_score, std_score

def subplots_centered(nrows, ncols, nfigs, figsize=(25,25)):
    """
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    
    fig = plt.figure(dpi=300, figsize=figsize)
    axs = []
    
    needs_center = nrows * ncols != nfigs

    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1 and needs_center: # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0
        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
        axs.append(ax)
        
    return fig, np.array(axs)

# Algorithm inspired by https://flothesof.github.io/farthest-neighbors.html
def incremental_farthest_search(points, k, solution_set):
    for _ in range(k):
        distances = [distance(p, solution_set[0]) for p in points]
        for i, p in enumerate(points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], distance(p, points[s]))
        solution_set.append(distances.index(max(distances)))
    return solution_set

def distance(p_1, p_2):
    return np.sum((p_1 - p_2) ** 2) ** (1/2)