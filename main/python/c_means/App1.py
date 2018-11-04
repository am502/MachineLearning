import numpy as np

A1 = np.array([[np.random.uniform(0, 20), np.random.uniform(0, 20)] for k in range(8)])
# print(A1)
# центры кластеров
A2 = np.array([[1, 1], [4, 6], [3, 5]])


# расстояние от каждой точки A1 до A2 (?)
def dist(list1, list2):
    return sum((i - j) ** 2 for i, j in zip(list1, list2))


dist = np.array([[dist(i, j) for i in A2] for j in A1])
# print(dist)
m = 1.1
# матрица принадлежности (?)
u = (1 / dist) ** (2 / (m - 1))
# [:, None] - поворот (?)
um = (u / u.sum(axis=1)[:, None])
# print(um)
# транспанирование для умножения
c = (um.T).dot(A1) / um.sum(axis=0)[:, None]
# матрица пересчитанных центров центроидов (?)
print(c)
