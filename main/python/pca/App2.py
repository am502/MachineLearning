import numpy as np

num = 13
dataset = np.array([[i, 2 * i + np.random.uniform(-3, 3)] for i in range(num)])

# print(dataset)

# plt.figure()
# plt.scatter(dataset[:, 0], dataset[:, 1], c='black')
# plt.show()

dataset_cntr = dataset - dataset.mean(axis=0)

# plt.figure()
# plt.scatter(dataset_cntr[:, 0], dataset_cntr[:, 1], c='green')
# plt.show()

covmat = np.cov(dataset_cntr, rowvar=False)

# первый список - собств. знач., второй - векторы ?
print(np.linalg.eig(covmat))

vals, vects = np.linalg.eig(covmat)

# Единичный вектор
# vect1 = vects[0]
# print(vect1[0] ** 2 + vect1[1] ** 2)

# без reshape-а (2,)
vect1 = vects[0].reshape(2, -1)
vect2 = vects[1].reshape(2, -1)

# Размерность матрицы
# print(dataset_cntr.shape)

print(np.dot(dataset_cntr, vect1))

coord1 = np.dot(dataset_cntr, vect1)
coord2 = np.dot(dataset_cntr, vect2)

# plt.figure()
# plt.scatter(coord1[:, 0], [0 for i in range(num)], c='red')
# plt.show()

# plt.figure()
# plt.scatter(coord2[:, 0], [0 for i in range(num)], c='blue')
# plt.show()

dot_prod = np.array([i[0] * vect1[0] + i[1] * vect1[1] for i in dataset_cntr])
print(dot_prod)
