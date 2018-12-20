import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import decomposition

iris = datasets.load_iris()
# features свойства
X = iris.data
# К какому классу относится ?
Y = iris.target

fig = plt.figure(1, figsize=(6, 5))
ax = Axes3D(fig, elev=48, azim=134)

# находит центроиды и ставит туда название ?
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Verginics', 2)]:
    ax.text3D(X[Y == label, 0].mean(),
              X[Y == label, 1].mean() + 1,
              X[Y == label, 2].mean(), name)

y_clr = np.choose(Y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_clr, cmap=plt.cm.nipy_spectral)

plt.show()

# sklearn method из коробки
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# .3 = 30% тестовые, 70% обуч.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=42)
# max depth - глубина графа
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, Y_train)
# далее используем тестовые данные для предсказания
pred = clf.predict_proba(X_test)

# print(Y_test)
# print(pred)

# макс по столбцам ?
print(pred.argmax(axis=1))

# в y test истинные значения, сравниваем
accuracy_score(Y_test, pred.argmax(axis=1))

# pca, уменьшили размерность с 4-х до 2-х, не 4 фичи, а 2
pca = decomposition.PCA(n_components=2)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered)
X_pca = pca.transform(X_centered)

# bo - синие точки
plt.plot(X_pca[Y == 0, 0], X_pca[Y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[Y == 1, 0], X_pca[Y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[Y == 2, 0], X_pca[Y == 2, 1], 'ro', label='Verginics')

plt.legend(loc=0)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, Y_train)
pred = clf.predict_proba(X_test)

# было 97% стало 95%, потеряли часть данных
print(accuracy_score(Y_test, pred.argmax(axis=1)))
