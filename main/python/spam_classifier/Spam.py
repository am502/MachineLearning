import math

spam = ["Путевки по низкой цене", "Акция! Купи шоколадку и получи телефон в подарок"]
imp = ["Завтра состоится собрание", "Купи килограмм яблок и шоколадку"]
test = ["В магазине гора яблок. Купи семь килограмм и шоколадку"]

noise = ['!', '.', '?', ',', ':']


def make_dict(list1):
    result = []
    for sent in list1:
        words = sent.split()
        for word in words:
            for n in noise:
                word = word.replace(n, '')
            if len(word) > 3:
                result += [word.lower()]
    return result


dataset = []
for item in make_dict(spam):
    dataset.append(([item], 0))
for item in make_dict(imp):
    dataset.append(([item], 1))


print(dataset)

def fit(dataset, alpha):
    classes, freq, tot = {}, {}, set()
    for feats, label in dataset:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
        for feat in feats:
            if (label, feat) not in freq:
                freq[(label, feat)] = 0
            freq[(label, feat)] += 1
        tot.add(tuple(feats))

    for label, feat in freq:
        freq[(label, feat)] = (alpha + freq[(label, feat)]) / (alpha * len(tot) + classes[label])
    for c in classes:
        classes[c] /= len(dataset)

    return classes, freq


def classify(classifier, features):
    classes, freq = classifier
    return min(classes.keys(), key=lambda cls: -math.log(classes[cls]) + sum(
        -math.log(freq.get((cls, feat), 10 ** (-7))) for feat in features))


classifier = fit(dataset, 0)
print(classify(classifier, make_dict(["Акция: КУПИ!"])))

from sklearn.externals import joblib

joblib.dump(classifier, 'data/dicts.pkl')

from sklearn.externals import joblib

classifier = joblib.load('data/dicts.pkl')
print(classifier)
