#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# 二值化的目的是为了简便
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img

def Train(trainset, train_labels):
    # 先验概率 P(Y=C_k)
    prior_probability = np.zeros(class_num) 
    # 条件概率 P(X^(j) = a_jl | Y=C_k) 
    # 维度为 (class_num,feature_len,2) 简单的概率乘法公式 条件概率的取值可能性为
    # 784 个特征, 10类结果, 每个特征取值为 0 1 两种情况 
    conditional_probability = np.zeros((class_num, feature_len, 2)) 
    N = len(train_labels)
    # 计算先验概率及条件概率
    for i in range(N):
        img = binaryzation(trainset[i]) 
        label = train_labels[i]
        # P(Y=C_k)
        prior_probability[label] += 1

        for j in range(feature_len):
            # 计算条件概率分子部分的
            conditional_probability[label][j][img[j]] += 1
    prior_probability = prior_probability/N*1.0
    # 将概率归到[0, 1.0]
    # 参考例子4-1的最后公式
    for i in range(class_num): # 在确定Y=c_k的条件概率
        for j in range(feature_len): # 特征取值

            # 经过二值化后图像只有0，1两种取值
            pix_0 = conditional_probability[i][j][0] # 在 Y=C_k 的情况下, X像素点为零的个数
            pix_1 = conditional_probability[i][j][1] # 在 Y=C_k 的情况下, X像素点为 1 的个数
             
            # 计算0，1像素点对应的条件概率
            probalility_0 = (float(pix_0 + 1.0)/float(pix_0+pix_1 + 2*1.0)) 
            probalility_1 = (float(pix_1 + 1.0)/float(pix_0+pix_1 + 2*1.0)) 

            conditional_probability[i][j][0] = probalility_0
            conditional_probability[i][j][1] = probalility_1

    return prior_probability,conditional_probability

# 计算概率
def calculate_probability(img, label):
    probability = prior_probability[label]
    for i in range(len(img)):
        probability *= conditional_probability[label][i][img[i]]

    return probability

def Predict(testset, prior_probability, conditional_probability):
    predict = []
    testset = binaryzation(testset)
    for img in testset:
        max_label = 0
        max_probability = calculate_probability(img, 0)

        for j in range(1, 10):
            probability = calculate_probability(img,j)

            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)

    return np.array(predict)


class_num = 10
feature_len = 784


# 仿照例子 4-1
'''
--------------------------------------------------------------
      (样本数) 1       2      3      4     5 ....        
--------------------------------------------------------------
X(1)  (特征取值)0/1
--------------------------------------------------------------
..
..
..
--------------------------------------------------------------
X(783)
--------------------------------------------------------------

'''
if __name__ == '__main__':

    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=66)
    print(train_features.shape)
    print(test_features.shape)

    time_2 = time.time()
    print('read data cost ',time_2 - time_1,' second','\n')

    print('Start training')
    prior_probability, conditional_probability = Train(train_features,train_labels)
    time_3 = time.time()
    print('training cost ',time_3 - time_2,' second','\n')

    print('Start predicting')
    test_predict = Predict(test_features, prior_probability, conditional_probability)
    time_4 = time.time()
    print('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print("The accruacy socre is ", score)

'''结果
Start read data
read data cost  3.233480930328369  second 

Start training
training cost  18.757874727249146  second 

Start predicting
predicting cost  64.61274909973145  second 

The accruacy socre is  0.8396825396825397

'''
