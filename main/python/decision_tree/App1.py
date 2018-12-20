import matplotlib.pyplot as plt
import numpy as np

# Два признака: цвет, размер
dataset = [
    [['Зеленый', 3], 'Яблоко'],
    [['Желтый', 3], 'Яблоко'],
    [['Красный', 1], 'Виноград'],
    [['Красный', 1], 'Виноград'],
    [['Желтый', 3], 'Лимон']
]

header = [['цвет', 'размер'], 'метка']


def unique_vals(rows, col):
    return set([row[0][col] for row in rows])


# Уникальные элементы по цвету
print(unique_vals(dataset, 0))


def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


print(class_counts(dataset))


def is_numeric(value):
    return isinstance(value, (int, float))


print(is_numeric('Красный'))


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "%s %s %s?" % (header[0][self.column], condition, str(self.value))


print(Question(1, 3))

# Задается вопрос первой строке первому элементу: цвет зеленый
q = Question(0, 'Зеленый')
example = dataset[0][0]
print(q.match(example))


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row[0]):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


print(partition(dataset, Question(0, 'Красный')))


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_label = counts[label] / float(len(rows))
        impurity -= prob_label ** 2
    return impurity


print(gini(dataset))

test1 = [
    [[], 'Яблоко'],
    [[], 'Яблоко'],
    [[], 'Яблоко'],
]

print(gini(test1))


def info_gain(left, right, current):
    p = float(len(left)) / (len(left) + len(right))
    return current - p * gini(left) - (1 - p) * gini(right)


current = gini(dataset)
true_rows, false_rows = partition(dataset, Question(0, 'Зеленый'))
print(info_gain(true_rows, false_rows, current))


def find_best_split(rows):
    best_gain = 0
    best_question = None
    current = gini(rows)
    n_features = len(rows[0][0])
    for col in range(n_features):
        values = set([row[0][col] for row in rows])

        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current)
            if gain >= best_gain:
                best_gain = gain
                best_question = question

    return best_gain, best_question


print(find_best_split(dataset))


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing="--"):
    if isinstance(node, Leaf):
        print(spacing + "Предположение", node.predictions)
        return

    print(spacing + str(node.question))
    print(spacing + "--> Да:")
    print_tree(node.true_branch, "    ")
    print(spacing + "--> Нет:", "    ")
    print_tree(node.false_branch)


tree = build_tree(dataset)
print_tree(tree)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row[0]):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


print(classify(dataset[0], tree))
print(classify(dataset[1], tree))


# counts - словарь (?)
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + "%"
    return probs


print(print_leaf(classify(dataset[1], tree)))

test = [
    [['Зеленый', 3]],
    [['Желтый', 4]],
    [['Красный', 2]],
    [['Красный', 1]],
    [['Желтый', 3]]
]

for row in test:
    print("Класс: %s" % print_leaf(classify(row, tree)))
