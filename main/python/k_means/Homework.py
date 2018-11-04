import random

import numpy as np
from matplotlib import pyplot as plt


class K_Means():
    def __init__(self, dataset, n_clusters, option):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.option = option
        self.max_n_iter = 10
        self.tolerance = .01
        self.fitter = False
        self.labels = np.array([])
        self.centroids = self.get_random_centroids()

    def get_dist2(self, list1, list2):
        if self.option == 1:
            return np.math.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))
        elif self.option == 2:
            return sum((i - j) ** 2 for i, j in zip(list1, list2))
        elif self.option == 3:
            return sum(np.math.fabs(i - j) for i, j in zip(list1, list2))
        elif self.option == 4:
            return max(np.math.fabs(i - j) for i, j in zip(list1, list2))

    def get_random_centroids(self):
        set_of_indexes = set()
        while len(set_of_indexes) != self.n_clusters:
            set_of_indexes.add(random.randrange(0, len(self.dataset)))
        return np.array([self.dataset[i] for i in set_of_indexes], dtype='f')

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

    def draw(self):
        if len(self.dataset[0]) == 2 and self.n_clusters < 10:
            colors = np.array(
                ['#42b9f4', '#41f44a', '#a3952f', '#f4427a', '#68f441', '#0b94a0', '#2b1526', '#e00830', '#d2e00b'])
            plt.figure()
            plt.scatter(self.dataset[:, 0], self.dataset[:, 1], c=colors[self.labels])
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=100, c='black')
            plt.show()


dataset = np.array([[1, 9], [2, 5], [7, 9], [1, 6], [5, 8], [3, 4], [9, 4], [6, 9], [9, 1]])
test = K_Means(dataset, 4, 2)
test.fit()
test.draw()
