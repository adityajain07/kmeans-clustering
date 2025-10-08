import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

x,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)

# Plot the initial clusters
fig = plt.figure(0)
plt.grid(True)
plt.scatter(x[:,0],x[:,1])

# Randomly choose any three points as initial centroids
num_centroids = 3
centroids = x[np.random.choice(x.shape[0], num_centroids, replace=False)]
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=100, label='Initial Centroids')

# Other algorithm variables
iterations = 100
min_change = 0.01

for i in range(iterations):
    cluster_assign = [[] for _ in range(num_centroids)]
    max_centroid_diff = 0
    for point in x:
        # Assign to nearest clusters
        nearest_centroid_idx = 0
        min_dist = 1e8
        for idx, centroid in enumerate(centroids):
            dist = euclidean_distance(point, centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_centroid_idx = idx
        cluster_assign[nearest_centroid_idx].append(np.round(point,2))

    # Update the cluster centroids
    for idx, cluster_group in enumerate(cluster_assign):
        centroid_mean = np.mean(cluster_group, axis=0)
        diff = euclidean_distance(centroid_mean, centroids[idx])
        if diff > max_centroid_diff:
            max_centroid_diff = diff
        centroids[idx] = centroid_mean
        
    # Check for the change in centod  
    print(f"Completed with iteration {i+1}.")
    if max_centroid_diff < min_change:
        break

print("K-means clustering has converged!")
plt.scatter(centroids[:, 0], centroids[:, 1], color='orange', marker='o', s=100, label='Final Centroids')
plt.legend()
plt.title(f"After iteration {i+1}.")
plt.show()