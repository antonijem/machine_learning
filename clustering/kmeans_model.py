import numpy as np

class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidian_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, X, max_iterations=1000):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for i in range(max_iterations):
            y = []
            if i % (max_iterations // 10) == 0:
                print(f"Iteracija: {i}")
            for data_point in X:
                distance = KMeansClustering.euclidian_distance(data_point, self.centroids)
                cluster_num = np.argmin(distance)
                y.append(cluster_num)

            y = np.array(y)

            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 0.000001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y
