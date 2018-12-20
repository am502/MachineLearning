import numpy as np
from main.python.knn.Pygame import Pygame

pygame = Pygame("KNN")

dataset = pygame.get_data()


class k_NN():
    def __init__(self, dataset, k_nums=5):
        self.dataset = dataset
        self.k_nums = k_nums

    def get_dist(self, list1, list2):
        return np.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))

    def predict(self, data):
        dist = np.array([[self.get_dist(data, i[0]), i[1][0]] for i in self.dataset])
        sort = dist[dist[:, 0].argsort()][:self.k_nums]
        counts = {}
        for s in sort:
            if int(s[1]) not in counts:
                counts[int(s[1])] = 0
            counts[int(s[1])] += 1
        return max(counts.keys(), key=lambda cl: counts[cl])


test = k_NN(dataset, 10)
print(test.predict([330, 240]))
