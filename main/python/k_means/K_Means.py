import numpy as np
from matplotlib import pyplot as plt


class K_Means():
    def __init__(self, dataset, n_clusters=3):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.max_n_iter = 10
        self.tolerance = .01
        self.fitter = False
        self.labels = np.array([])
        self.centroids = np.array([self.dataset[k] for k in range(self.n_clusters)], dtype='f')
        # print(self.centroids)

    def get_dist2(self, list1, list2):
        return sum((i - j) ** 2 for i, j in zip(list1, list2))

    def distribute_data(self):
        self.labels = np.array([])
        for elem in self.dataset:
            dist2 = [self.get_dist2(elem, center) for center in self.centroids]
            idx = dist2.index(min(dist2))
            self.labels = np.append(list(self.labels), idx).astype(int)

    def recalculate_centroids(self):
        for i in range(self.n_clusters):
            num = 0
            temp = np.zeros(self.dataset[0].shape)
            for k, label in enumerate(self.labels):
                if label == i:
                    temp = temp + self.dataset[k]
                    num += 1
            self.centroids[i] = temp / num

        # print(self.centroids)

    def fit(self):
        iter = 1;
        while iter < self.max_n_iter:
            prev_centroids = np.copy(self.centroids)
            self.distribute_data()
            self.recalculate_centroids()
            if max([self.get_dist2(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                break
            iter += 1
        self.fitted = True

    def predict(self, data):
        if self.fitted:
            dist2 = [self.get_dist2(data, center) for center in self.centroids]
            return dist2.index(min(dist2))


l1 = np.array([[1, 2], [3, 4], [5, 6], [8, 3], [7, 2], [8, 8], [1, 1], [5, 1]])

test = K_Means(l1, 3)
test.distribute_data()
test.recalculate_centroids()

test.fit()
print(test.predict([0, 4]))

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])

plt.figure()
plt.scatter(test.dataset[:, 0], test.dataset[:, 1], c=colors[test.labels])
plt.scatter(test.centroids[:, 0], test.centroids[:, 1], marker='x', s=100, c='black')
plt.show()
