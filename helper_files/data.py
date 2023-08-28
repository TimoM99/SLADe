import numpy as np
import math
import random

from sklearn.datasets import make_moons


def make_2_moons_data(nb_anoms, nb_normals, dist_between_normal_anom, random_state):
    data = make_moons(n_samples=nb_normals, noise=0.1, random_state=random_state)
    normals = data[0]
    np.random.seed(random_state)
    x_anom = list(np.random.uniform(-2, 2, nb_anoms))
    y_anom = list(np.random.uniform(-2, 2, nb_anoms))
    anoms = zip(x_anom, y_anom)
    anoms = np.array(list(filter(
        lambda x: not any(map(
            lambda y: math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2) < dist_between_normal_anom, 
            data[0])), 
        anoms)))
    return anoms, normals

def make_overlap_dt_data(nb_anoms, nb_normals, random_state):
    np.random.seed(random_state)
    normals = np.random.multivariate_normal([0, 0], [[1/2, 0],[0, 1/2]], size=nb_normals)
    anoms = np.random.multivariate_normal([1, 1], [[1/2, 0], [0, 1/2]], size=nb_anoms)

    return anoms, normals

def create_unlabeled_dataset(anoms, normals):
    X_data = [*normals, *anoms]
    y_data = [0] * len(X_data)
    return X_data, y_data

def create_fully_labeled_dataset(anoms, normals):
    fully_labeled_X = [*normals, *anoms]
    random.Random(331).shuffle(fully_labeled_X)
    fully_labeled_y = []
    for X in fully_labeled_X:
        if np.any(np.all(X == anoms, axis=1)):
            fully_labeled_y.append(1)
        else:
            fully_labeled_y.append(-1)
    
    return np.array(fully_labeled_X), np.array(fully_labeled_y)
