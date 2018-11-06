import math

from nltk.stem.snowball import SnowballStemmer

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


def classify(classifier, features):
    classes, freq = classifier
    return min(classes.keys(), key=lambda cls: -math.log(classes[cls]) + sum(
        -math.log(freq.get((cls, feat), 10 ** (-7))) for feat in features))


examples = ["Сегодня заработай, завтра выведи! 4 часа работы - заработок 38,273 рублей. Узнай как!",
            "Обязательно приходите на собрание", "Купи килограмм яблок и шоколадку со скидкой",
            "Добрый день. Высылаю вопросы, зачет в пятницу.",
            "Завтра в 17:00 всем обязательно быть в 8 утра в университете!"]

from sklearn.externals import joblib

classifier = joblib.load('homework_data/dicts.pkl')
print(classifier)

for text in examples:
    print(classify(classifier, make_dict(text)))
