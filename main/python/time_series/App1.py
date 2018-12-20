import matplotlib.pyplot as plt
import numpy as np

dataset = np.empty((0, 2))

with open("sunspot.txt", "r") as file:
    data = file.readlines()
    for i, line in enumerate(data):
        if 2 < i < 3079:
            words = line.split()
            if len(words) > 1:
                dataset = np.append(dataset, [[i - 2, float(words[-1])]], axis=0)

print(dataset)

x = dataset[:, 0]
y = dataset[:, 1]

plt.figure(figsize=(20, 10))
plt.xlim(0, 1000)
plt.scatter(x, y)
plt.plot(x, y)
plt.show()


# windowszie - пред. элементы (?)
def moving_average(series, windowsize):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(sum(series[n - i] for i in range(windowsize)) / float(windowsize))
    return result


plt.figure(figsize=(20, 10))
plt.xlim(0, 1000)
plt.scatter(x, y)
y_avg = moving_average(y, 10)
plt.plot(x, y_avg, c='red', linewidth=5)
plt.show()


def weighted_average(series, weights):
    results = [series[0]]
    for n in range(1, len(series)):
        result = 0.0
        for k in range(len(weights)):
            result += series[n - k - 1] * weights[k]
        results.append(result)
    return results


plt.figure(figsize=(20, 10))
plt.xlim(0, 1000)
plt.scatter(x, y)
y_wv = weighted_average(y, [.6, .2, .1, .07, .03])
plt.plot(x, y_wv, c='red', linewidth=5)
plt.show()


def exp_smoothing(series, alpha):
    results = [series[0]]
    for n in range(1, len(series)):
        results.append(alpha * series[n] + (1 - alpha) * results[n - 1])
    return results


plt.figure(figsize=(20, 10))
plt.xlim(0, 1000)
plt.scatter(x, y)
# y_es = exp_smoothing(y, 1)
y_es = exp_smoothing(y, .3)
plt.plot(x, y_es, c='red', linewidth=5)
plt.show()


def dexp_smoothing(series, alpha, beta):
    results = []
    for n in range(1, len(series)):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        else:
            last_level, level = level, alpha * series[n] + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
        results.append(level + trend)
    return results


plt.figure(figsize=(20, 10))
plt.xlim(0, 1000)
plt.scatter(x, y)
y_es = dexp_smoothing(y, .01, .2)
plt.plot(x[:len(y_es)], y_es, c='red', linewidth=5)
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
plt.xlim(0, 1000)
plt.scatter(x, y)
y_r, y_l, y_t = dexp_smoothing_2(y, .6, .6)
plt.plot(x[:len(y_es)], y_r, c='red', linewidth=5, label="result")
plt.plot(x[:len(y_es)], y_l, c='green', linewidth=5, label="level")
plt.plot(x[:len(y_es)], y_t, c='blue', linewidth=5, label="trend")
plt.legend(loc='best')
plt.show()


# slen = season length, длина серии (?)
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i + slen] - series[i]) / slen
    return sum / slen


def initial_seasonal_comp(series, slen):
    seasonals = {}
    season_avg = []
    n_seasons = int(len(series) / slen)
    for j in range(n_seasons):
        season_avg.append(sum(series[slen * j:slen * j + slen]) / float(slen))
    for i in range(slen):
        sum_of_values_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_values_over_avg += series[slen * j + i] - season_avg[j]
        seasonals[i] = sum_of_values_over_avg / n_seasons
    return seasonals


def texp_smoothing(series, slen, alpha, beta, gamma):
    results = []
    levels = []
    trends = []
    seasons = {}
    for i in range(slen):
        seasons[i] = np.zeros((0, 2))
    seasonals = initial_seasonal_comp(series, slen)
    for i in range(len(series)):
        if i == 0:
            level = series[0]
            trend = initial_trend(series, slen)
            results.append(series[0])
            continue
        else:
            last_level, level = level, alpha * (series[i] - seasonals[i % slen]) + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            seasonals[i % slen] = gamma * (series[i] - level) + (1 - gamma) * seasonals[i % slen]
        results.append(level + trend + seasonals[i % slen])
        levels.append(level)
        trends.append(trend)
        seasons[i % slen] = np.append(seasons[i % slen], [[i, seasonals[i % slen]]], axis=0)
    return results, levels, trends, seasons


plt.figure(figsize=(20, 10))
plt.xlim(0, 1000)
plt.scatter(x, y)
y_r, y_l, y_t, y_s = texp_smoothing(y, 11, .01, .2, .1)
# Кол-во элементов поправить
plt.plot(x[:len(y_r)], y_r, c='red', linewidth=5)
plt.plot(x[:len(y_r)], y_l, c='green', linewidth=5)
plt.plot(x[:len(y_r)], y_t, c='blue', linewidth=5)
plt.legend(loc='best')
plt.show()
