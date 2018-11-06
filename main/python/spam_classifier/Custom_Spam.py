import math

from nltk.stem.snowball import SnowballStemmer

# print(ord('а'))
# print(ord('я'))

ham_file = open("homework_data/ham.txt", "r")
spam_file = open("homework_data/spam.txt", "r")

with ham_file as file:
    ham = file.read()

with spam_file as file:
    spam = file.read()

stemmer = SnowballStemmer("russian")


def make_dict(text):
    result = []
    words = text.split()
    for word in words:
        curWord = ''
        for char in word:
            if 1072 <= ord(char.lower()) <= 1103:
                curWord += char
        if len(curWord) > 3:
            result += [stemmer.stem(curWord.lower())]
    return result


dataset = []
for word in make_dict(spam):
    dataset.append(([word], 0))
for word in make_dict(ham):
    dataset.append(([word], 1))

print(len(dataset))


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
print(classify(classifier, make_dict("Завтра в 17:00 всем обязательно быть!")))

from sklearn.externals import joblib

joblib.dump(classifier, 'homework_data/dicts.pkl')