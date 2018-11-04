import numpy as np
from matplotlib import pyplot as plt

from main.python.db_scan.DB_Scan import DB_SCAN
from main.python.pygame.Pygame import Pygame

A1 = {0: [1, 2, 3], 1: [4, 5, 6], 3: [7, 8, 9]}
print(A1[0])
A1[4] = [10, 11, 12]
print(A1)

A2 = set()
A2.add(1)
A2.add(2)
A2.add(1)
print(A2)

A3 = np.array([5, 6, 7])
list(A3)
tuple(A3)

# нельзя изменять A4[0] = 9
A4 = (4, 4)
print(A4[0])

pygame = Pygame("DB Scan")
dataset = pygame.get_data()

colors = np.array(['#f442c5', '#d1060d', '#631899', '#ff00d4'])

test = DB_SCAN(dataset, 30, 2)
pred = test.get_labels()

plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[pred])
plt.show()

# график зависимости ?
n_clust = []
for k in range(1, 20):
    test = DB_SCAN(dataset, k * 5, 2)
    test.fit()
    n_clust.append(test.n_clusters)
    print(test.n_clusters)

plt.plot(range(1, 20), n_clust, marker='s')
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$')
