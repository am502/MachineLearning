import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
kmeans.predict([[0, 0], [4, 4]])
kmeans.cluster_centers_

#

kmeans.cluster_centers_[kmeans.labels_]

#

from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])

X_1, _ = make_blobs(n_samples=300, random_state=42)
kmeans = KMeans(n_clusters=3).fit(X_1)
pred = kmeans.fit_predict(X_1)
centroids = kmeans.cluster_centers_

#

plt.figure()
plt.scatter(X_1[:, 0], X_1[:, 1], c=colors[pred])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='black')
plt.show()

#

from sklearn.datasets import make_circles
from matplotlib import pyplot as plt

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])

X_1, _ = make_circles(n_samples=300, factor=0.5, noise=0.08)
kmeans = KMeans(n_clusters=3).fit(X_1)
pred = kmeans.fit_predict(X_1)
centroids = kmeans.cluster_centers_

plt.figure()
plt.scatter(X_1[:, 0], X_1[:, 1], c=colors[pred])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='black')
plt.show()

#

from matplotlib.image import imread


def get_img(img_path, show=True):
    orig_img = imread(img_path)

    if show:
        plt.imshow(orig_img)
        plt.show()
        print('Shape: ', orig_img.shape)

    return orig_img


get_img('images/01.png')


#

def get_kmeans(orig_img, n_colors=8):
    X = orig_img.reshape((-1, 4))
    kmeans = KMeans(n_clusters=n_colors).fit(X)
    pred = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    return centroids[pred].reshape(orig_img.shape)


#

n_colors = 4
all_img = []
orig_img = get_img('images/01.png', show=False)
new_img = get_kmeans(orig_img, n_colors)
all_img += [orig_img]
all_img += [new_img]

fig, axarr = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(20, 8))
axarr[0].imshow(all_img[0])
axarr[1].imshow(all_img[1])
fig.tight_layout()
plt.show()
