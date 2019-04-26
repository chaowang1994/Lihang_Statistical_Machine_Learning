#coding=utf-8
import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self, learning_step = 0.00001, max_iteration = 5000):
        self.learning_step = learning_step
        self.max_iteration = max_iteration

    def predict_(self, x):
        # 对应元素相乘 element-wise product: np.multiply(), 或 *
        wx = sum(self.w*x)
        return int(wx > 0)

    def train(self, features, labels):
        self.w = np.array([0.0] * (len(features[0]) + 1))
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum(self.w * x) # np 的元素 elsewise 相乘

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            self.w += np.dot((self.learning_step * y), x) # np.dot 矩阵乘法 *np的对应位置相乘

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels

    def predict_(self, x):
        # 对应元素相乘 element-wise product: np.multiply(), 或 *
        wx = sum(self.w*x)
        return int(wx > 0)


if __name__ == '__main__':

    time_1 = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print('Read data: ', time_2 - time_1, ' second!', '\n')

    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print('train time: ', time_3 - time_2, ' second', '\n')

    test_predict = p.predict(test_features)
    time_4 = time.time()
    print('predict time: ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print("The accuracy: ", score)


'''
我实现的 numpy 的乘法,即使用矩阵乘法,有加速作用:

    操作      原作者      我的矩阵版本(秒)
read data    3.2203       2.72776

training     1.48296      1.068635

predicting   3.45773      2.708252
 
accruacy     0.96507      0.9806637

'''