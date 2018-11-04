import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

from main.python.pygame.Pygame import Pygame

pygame = Pygame("DB Scan")
dataset = pygame.get_data()

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#000000'])

# Default DB scan
pred = DBSCAN(eps=40, min_samples=2).fit_predict(dataset)

plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[pred])
plt.show()

#

colors = np.array(['#f442c5', '#d1060d', '#631899', '#ff00d4'])

pred = DBSCAN(eps=40, min_samples=2).fit_predict(dataset)

plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[pred])
plt.show()
