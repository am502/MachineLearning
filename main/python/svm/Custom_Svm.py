import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1]
])

Y = np.array([-1, -1, 1, 1, 1])


# Чем меньше eta, тем точнее, но должно быть больше шагов (epochs)
def svm_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = .1
    epochs = 10000
    errors = []

    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            if (Y[i] * np.dot(X[i], w)) < 1:
                w += eta * (-2 * (1 / epoch) * w + X[i] * Y[i])
                error = 1
            else:
                w += eta * (-2 * (1 / epoch) * w)
            errors.append(error)

    plt.plot(errors, '|')
    plt.ylim(.5, 1.5)
    plt.show()

    return w


w = svm_sgd(X, Y)

#

color = np.array(['red', 'green', 'blue', 'yellow'])
plt.scatter(X[:, 0], X[:, 1], c=color[Y], marker='+')

x_min, x_max = X[:, 0].min(), X[:, 0].max()
# Не имеет отношения к лейблам (признакам) (?)
y_min, y_max = ((w[2] - w[0] * x_min) / w[1], (w[2] - w[0] * x_max) / w[1])

plt.plot([x_min, x_max], [y_min, y_max])
plt.show()
