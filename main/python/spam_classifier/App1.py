# file = open("data/test.txt", "w")
# file.write("First line\n")
# file.write("Second line")
# file.close()

file1 = open("data/m+.txt", "w")
file2 = open("data/m.txt", "r", encoding='utf-8')

for line in file2:
    newline = line[: -1] + ",0\n"
    # print(newline)
    file1.write(newline)

file3 = open("data/f+.txt", "w")
file4 = open("data/f.txt", "r", encoding='utf-8')

for line in file4:
    newline = line[: -1] + ",1\n"
    # print(newline)
    file3.write(newline)

file1.close()
file2.close()
file3.close()
file4.close()

import numpy as np

sample = np.empty((0, 2))
with open("data/m+.txt", "r") as file1:
    data = file1.readlines()
    for line in data:
        words = line.split(',')
        # print(words)
        sample = np.append(sample, [[words[0], words[1][0]]], axis=0)

with open("data/f+.txt", "r") as file1:
    data = file1.readlines()
    for line in data:
        words = line.split(',')
        # print(words)
        sample = np.append(sample, [[words[0], words[1][0]]], axis=0)

# print(sample)
dataset = np.random.permutation(sample)
# print(dataset)

# берем последнюю букву
features = [(feat[-1], int(label)) for feat, label in dataset]


# print(features)

def fit(dataset):
    classes, freq = {}, {}
    for feats, label in dataset:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
        for feat in feats:
            if (label, feat) not in freq:
                freq[(label, feat)] = 0
            freq[(label, feat)] += 1

    for label, feat in freq:
        freq[(label, feat)] /= classes[label]
    for c in classes:
        classes[c] /= len(dataset)

    return classes, freq


# fit(features)

classifier = fit(features)

import math


def classify(classifier, features):
    classes, freq = classifier
    return min(classes.keys(), key=lambda cls: -math.log(classes[cls]) + sum(
        -math.log(freq.get((cls, feat), 10 ** (-7))) for feat in features))


classify(classifier, "i")


def get_features(sample):
    features = sample[0] + sample[-2] + sample[-1]
    return features.lower()


# get_features("Геннадий")

features = [([get_features(feat)], int(label)) for feat, label in dataset]
classifier = fit(features)
# print(classifier)
print(classify(classifier, [get_features("Arturia")]))
