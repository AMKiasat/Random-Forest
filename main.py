import numpy as np
import random
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


if __name__ == '__main__':
    data, label = reading_files('Breast Cancer dataset/Breast_Cancer_dataset.txt')
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=1)
