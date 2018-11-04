import numpy as np


class DB_SCAN():
    def __init__(self, dataset, eps=3, min_samples=2):
        self.dataset = dataset
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = 0
        # бесхозный кластер
        self.clusters = {0: []}
        self.visited = set()
        self.clustered = set()
        self.fitted = False

    def get_dist(self, list1, list2):
        return np.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))

    # список соседей точки
    def get_region(self, data):
        return [list(q) for q in self.dataset if self.get_dist(data, q) < self.eps]

    def fit(self):
        for p in self.dataset:
            if tuple(p) in self.visited:
                continue
            self.visited.add(tuple(p))
            neighbours = self.get_region(p)
            if len(neighbours) < self.min_samples:
                self.clusters[0].append(list(p))
            else:
                self.n_clusters += 1
                self.expand_cluster(p, neighbours)
        self.fitted = True

    def expand_cluster(self, p, neighbours):
        if self.n_clusters not in self.clusters:
            self.clusters[self.n_clusters] = []
        self.clustered.add(tuple(p))
        self.clusters[self.n_clusters].append(list(p))
        while neighbours:
            q = neighbours.pop()
            if tuple(q) not in self.visited:
                self.visited.add(tuple(q))
                q_neighbours = self.get_region(q)
                if len(q_neighbours) > self.min_samples:
                    neighbours.extend(q_neighbours)
            if tuple(q) not in self.clustered:
                self.clustered.add(tuple(q))
                self.clusters[self.n_clusters].append(q)
                if q in self.clusters[0]:
                    self.clusters[0].remove(q)

    def get_labels(self):
        labels = np.array([])
        if not self.fitted:
            self.fit()
        for data in self.dataset:
            for i in range(self.n_clusters + 1):
                if list(data) in self.clusters[i]:
                    labels = np.append(labels, i).astype(int)

        return labels
