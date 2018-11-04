import numpy as np
from matplotlib import pyplot as plt

from main.python.pygame.Pygame import Pygame


class C_Means():
    def __init__(self, dataset, n_clusters=3, m=2, cut_param=.9):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.m = m
        self.cut_param = cut_param
        self.max_n_iter = 100
        self.tolerance = .01
        self.dist = np.zeros((self.dataset.shape[0], self.n_clusters))
        self.centroids = np.zeros((self.n_clusters, self.dataset.shape[1]))
        self.u = np.array(
            [[np.random.uniform(0, 1) for i in range(self.n_clusters)] for j in range(self.dataset.shape[0])])

    def get_dist2(self, list1, list2):
        return sum((i - j) ** 2 for i, j in zip(list1, list2))

    def distribute_data(self):
        self.dist = np.array([[self.get_dist2(i, j) for i in self.centroids] for j in self.dataset])
        self.u = (1 / self.dist) ** (2 / (self.m - 1))
        self.u = (self.u / self.u.sum(axis=1)[:, None])

    def recalculate_centroids(self):
        self.centroids = (self.u.T).dot(self.dataset) / self.u.sum(axis=0)[:, None]

    def fit(self):
        iter = 1;
        while iter < self.max_n_iter:
            prev_centroids = np.copy(self.centroids)
            self.recalculate_centroids()
            self.distribute_data()
            if max([self.get_dist2(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                break
            iter += 1

    def get_labels(self):
        l1 = self.u.argmax(axis=1)
        l2 = [i + 1 for i in l1]
        return np.array([j if i > self.cut_param else 0 for i, j in zip(self.u.max(axis=1), l2)])


pygame = Pygame("C means")
dataset = pygame.get_data()

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])
test = C_Means(dataset, 2, 1.1, 0.9)
test.fit()
pred = test.get_labels()

plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[pred])
plt.show()
