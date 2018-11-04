import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from main.python.pygame.Pygame import Pygame

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])

pygame = Pygame("K means")
dataset = pygame.get_data()

print(dataset.shape)
test = KMeans(n_clusters=3).fit_predict(dataset)

plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[test])
plt.show()
