import numpy as np
from sklearn.metrics import pairwise_distances


def uncertainty_score(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(x, centroids)
    two_closest_centroids = np.argsort(distances, axis=1)[:, :2]
    closest_distances = np.take_along_axis(distances, two_closest_centroids, axis=1)
    uncertainty_score = closest_distances[:, 0] / closest_distances[:, 1]
    return uncertainty_score
