import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1]
])

Y = np.array([0, 0, 1, 1, 1])


def logreg(X, Y):
    w = np.zeros(len(X[0]))
    eta = .01
    epochs = 10000

    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            w += eta * (Y[i] - 1 / (1 + np.exp(-np.dot(X[i], w)))) * X[i]
    return w


w = logreg(X, Y)

#

color = np.array(['red', 'green', 'blue', 'yellow'])
plt.scatter(X[:, 0], X[:, 1], c=color[Y], marker='+')

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = ((w[2] - w[0] * x_min) / w[1], (w[2] - w[0] * x_max) / w[1])

plt.plot([x_min, x_max], [y_min, y_max])
plt.show()
