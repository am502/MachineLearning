from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles

X_1, _ = make_circles(n_samples=300, factor=0.5, noise=0.08)

inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X_1)
    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, 8), inertia, marker='s')
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$')

#

from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D

iris = load_iris()
X = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3).fit(X)
pred = kmeans.fit_predict(X)

fig = plt.figure()
ax = Axes3D(fig)

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])

plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors[pred])
plt.show()
