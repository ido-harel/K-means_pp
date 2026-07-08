import sys
import numpy as np
import pandas as pd
import math
DEFAULT_K = 3

def print_and_exit(message):
    print(message)
    sys.exit(1)

# K-means++ functions
def euclidean_distances_from_centroids(data_points, centroids):
    dist = np.linalg.norm(
        data_points[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )
    min_dist = np.min(dist, axis=1)
    return min_dist

def probabilities(min_dist, chosen_indices):
    min_dist = min_dist.copy()

    for index in chosen_indices:
        min_dist[index] = 0.0
    total = np.sum(min_dist)

    probabilities = np.zeros(len(min_dist))
        unchosen_indices = [
            i for i in range(len(min_dist)) if i not in chosen_indices
        ]
        for i in unchosen_indices:
            probabilities[i] = 1.0 / len(unchosen_indices)
        return probabilities

    return min_dist / total

def choose_next_centroid(probabilities):
    N = len(probabilities)
    return np.random.choice(N, p=probabilities)

def kmeans_pp_init(data_points, K):
    np.random.seed(1234)
    N = data_points.shape[0]
    first_index = np.random.choice(N)

    chosen_indices = [first_index]
    centroids = [data_points[first_index]]

    while len(chosen_indices) < K:
        current_centroids = np.array(centroids)

        min_dist = euclidean_distances_from_centroids(data_points,current_centroids)
        probabilities = probabilities(min_dist,chosen_indices)
        next_index = choose_next_centroid(probabilities)

        chosen_indices.append(next_index)
        centroids.append(data_points[next_index])

    return chosen_indices, np.array(centroids)

def load_data(file1, file2):
   
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    merged = pd.merge(df1, df2, on=0, how="inner")
    merged = merged.sort_values(by=0).reset_index(drop=True)
    data_points = merged.iloc[:, 1:].to_numpy(dtype=float)

    return data_points

def parse_args():
    len = len(sys.argv)

    if len == 6:
        k = sys.argv[1]
        iter = sys.argv[2]
        eps = sys.argv[3]
        file1 = sys.argv[4]
        file2 = sys.argv[5]

        if not k.isdigit():
            print_and_exit("Incorrect number of clusters!")
        K = int(k)

    elif len == 5:
        K = DEFAULT_K
        iter = sys.argv[1]
        eps = sys.argv[2]
        file1 = sys.argv[3]
        file2 = sys.argv[4]

    else:
        print_and_exit("An Error Has Occurred")

    # The upper bound K < N is checked later, after loading the data.
    if K <= 1:
        print_and_exit("Incorrect number of clusters!")

    if not iter.isdigit():
        print_and_exit("Incorrect maximum iteration!")

    iter = int(iter)

    if not (1 < iter < 400):
        print_and_exit("Incorrect maximum iteration!")

    try:
        eps = float(eps)
    except ValueError:
        print_and_exit("Incorrect epsilon!")

    if not math.isfinite(eps) or eps < 0:
        print_and_exit("Incorrect epsilon!")

    return K, iter, eps, file1, file2

def main():
    K, iter, eps, file1, file2 = parse_args()
    data_points = load_data(file1, file2)
    N = data_points.shape[0]

    if not (K < N):
        print_and_exit("Incorrect number of clusters!")

    indices, initial_centroids = kmeans_pp_init(data_points, K)

    # later we wiil need the wrapper function 
    # final_centroids = mykmeanssp.py_run_kmeans(data_points, initial_centroids, K, iter, eps)