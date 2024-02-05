import numpy as np
import random
import math
import pickle
from scale import scale_into_number
from sklearn.model_selection import train_test_split


def reading_files(filename):
    list1 = []
    list2 = []

    with open(filename, 'r') as file:
        for line in file:
            values = [value.strip("'") for value in line.strip().split(',')]
            if values.__contains__('?'):
                continue
            scaled = scale_into_number(values)
            list2.append(scaled.pop())
            list1.append(scaled)
    data = np.array(list1, dtype=int)
    label = np.array(list2)
    return data, label


def entropy_calculator(x):
    entropy = 0
    for i in x:
        if i > 0:
            entropy += -1 * (i * math.log2(i))
    return entropy


def spliter(feature, threshold):
    left_indices = np.where(feature <= threshold)[0]
    right_indices = np.where(feature > threshold)[0]
    return left_indices, right_indices


def label_counter(indices, y):
    if len(indices) > 0:
        y0 = np.where(y == -1)
        y1 = np.where(y == 1)
        count0 = 0
        count1 = 0
        count2 = 0
        for i in indices:
            if i in y0[0]:
                count0 += 1
            elif i in y1[0]:
                count1 += 1
        return [count0 / len(indices), count1 / len(indices)]
    else:
        return [0]


def find_best_split(x, y, y_entropy):
    best_gain_ratio = [0, 0, 0]  # [gain_ratio, feature, value]
    for feature in range(len(x.T)):
        selected_spliter = []
        for value in x.T[feature]:
            # print(value)
            if value not in selected_spliter:
                selected_spliter.append(value)
                left_indices, right_indices = spliter(x.T[feature], value)
                entropy = len(left_indices) * entropy_calculator(label_counter(left_indices, y))
                entropy += len(right_indices) * entropy_calculator(label_counter(right_indices, y))
                entropy /= len(y)
                gain = y_entropy - entropy
                split = entropy_calculator([len(left_indices) / len(y), len(right_indices) / len(y)])
                if split > 0:
                    gain_ratio = gain / split
                else:
                    gain_ratio = gain
                if gain_ratio > best_gain_ratio[0]:
                    best_gain_ratio = [gain_ratio, feature, value]
                    # print(best_gain_ratio)
                # elif gain_ratio == best_gain_ratio[0]:
                #     print([gain_ratio, feature, value])
    return best_gain_ratio


def grow_tree(x, y, offset_num):
    size = len(y)
    label_count = [np.count_nonzero(y == -1) / size,
                   np.count_nonzero(y == 1) / size]
    lc = [np.count_nonzero(y == -1), np.count_nonzero(y == 1)]
    # print(y)
    if size - np.max(lc) <= offset_num:
        # print(size, np.max(lc))
        return np.argmax(lc)
    best_split = find_best_split(x, y, entropy_calculator(label_count))
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for i in range(len(x)):
        if x[i][best_split[1]] <= best_split[2]:
            left_x.append(x[i])
            left_y.append(y[i])
        else:
            right_x.append(x[i])
            right_y.append(y[i])
    if len(right_y) == 0:
        return left_y[0]
    elif len(left_y) == 0:
        return right_y[0]
    # print(best_split, len(left_x), len(right_x))
    left = grow_tree(np.array(left_x), np.array(left_y), offset_num)
    right = grow_tree(np.array(right_x), np.array(right_y), offset_num)

    return left, best_split[1], best_split[2], right


def train(data, label, offset_num, tree_num):
    forest = []
    feature_num = len(data.T)
    sub_feature_num = int(math.log2(feature_num) + 1)
    for i in range(tree_num):
        selected_features = random.sample(range(len(data.T)), sub_feature_num)
        selected_features = np.sort(selected_features)
        new_data, _, new_label, _ = train_test_split(data, label, test_size=0.2, random_state=1)
        selected_columns = new_data[:, selected_features]
        # print(selected_features)
        # for i in range(len(selected_columns)):
        #     print(selected_columns[i], new_data[i])
        tree = grow_tree(selected_columns, new_label, offset_num=offset_num)
        print(i, tree)
        forest.append(tree)

    with open('forest.pkl', 'wb') as file:
        pickle.dump(forest, file)


def predict(x, tree):
    if isinstance(tree, np.int64):
        return tree
    elif x[tree[1]] <= tree[2]:
        return predict(x, tree[0])
    else:
        return predict(x, tree[3])


if __name__ == '__main__':
    data, label = reading_files('Breast Cancer dataset/Breast_Cancer_dataset.txt')
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=1)
    train(train_data, train_labels, 0, 20)
