# encoding=utf-8
# @Author: wangchao
# @Date:   11-24-2018
# @Email:  chao.wanghs@gmail.com
# @Last modified by:   wangchao
# @Last modified time: 11-24-2018


import pandas as pd
import numpy as np
import cv2
import random
import time

# 数据是手写字体 28*28 的
# 格式为： label,pixel0,pixel1,pixel2, ..., pixel782,pixel783 
 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.0001
        self.max_iteration = 5000

    def predict(self, x):
        wx = sum([self.w[j] * x[j] for j in xrange(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        #self.w = [0.0] * len(features[0])
        mu,sigma=0, 0.1 #均值与标准差
        self.w = np.random.normal(mu,sigma,len(features[0])) 
        self.b = 1.0
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in xrange(len(self.w))])+self.b 
            time += 1
            if wx * y > 0:
                correct_count += 1
            else:
                for i in xrange(len(self.w)):
                    self.w[i] += self.learning_step * (y * x[i])
                self.b += self.learning_step * y 

    def predict_batch(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            labels.append(self.predict(x))
        return labels


if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::] # 行,列
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=1./3, random_state=666)

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print ('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ', time_3 - time_2, ' second', '\n')

    print ('Start predicting')
    test_predict = p.predict_batch(test_features)
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)
