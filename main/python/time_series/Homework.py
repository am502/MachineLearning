import matplotlib.pyplot as plt
import numpy as np

file = open("30marsh.txt", "r")

dataset = np.empty((0, 2))

with file as file:
    data = file.readlines()
    for i, line in enumerate(data):
        if line[0] != '#':
            number = float(line.split(' ')[2])
            dataset = np.append(dataset, [[i, number]], axis=0)

print(dataset)

x = dataset[:, 0]
y = dataset[:, 1]

plt.figure(figsize=(20, 10))
plt.xlim(0, 700)
plt.scatter(x, y)
plt.plot(x, y)
plt.show()


def dexp_smoothing_2(series, alpha, beta):
    results = []
    levels = []
    trends = []
    for n in range(1, len(series)):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        else:
            last_level, level = level, alpha * series[n] + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
        results.append(level + trend)
        levels.append(level)
        trends.append(trend)
    return results, levels, trends


plt.figure(figsize=(20, 10))
plt.xlim(0, 700)
plt.scatter(x, y)
y_r, y_l, y_t = dexp_smoothing_2(y, .4, .2)
plt.plot(x[:len(y_r)], y_r, c='red', linewidth=5, label="result")
plt.plot(x[:len(y_r)], y_l, c='green', linewidth=5, label="level")
plt.plot(x[:len(y_r)], y_t, c='blue', linewidth=5, label="trend")
plt.legend(loc='best')
plt.show()
