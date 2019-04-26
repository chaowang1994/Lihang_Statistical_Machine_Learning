#coding:utf-8
import numpy as np
import itertools
from scipy.stats import mode
import sys
sys.setrecursionlimit(1000000)
import copy
import struct
import time

# 参考大佬:https://github.com/qingshangithub/KNNbyMatlab

train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

# mnist 格式参考: http://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file,'rb').read()

    #解析文件头信息 
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('图片数量：%d，图片大小：%d*%d' %(num_images, num_rows, num_cols))
    #解析数据集
    image_size = num_cols * num_rows
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        #if (i+1)%10000 == 0:
            #print("已解析 %d" %(i + 1) + "张")
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):

    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('图片数量: %d张' % (num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        #if (i + 1) % 10000 == 0:
            #print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file = train_images_path):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file = train_labels_path):
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file = test_images_path):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file = test_labels_path):
    return decode_idx1_ubyte(idx_ubyte_file)



class KDNode: #KD树的节点

    def __init__(self, point, dim):
        self.point = point  # kd树储存的点
        self.split_dim = dim  # 分割维度
        self.left = None
        self.right = None


def findMedian(data_lst, split_dim):
    '''
    找出 data_lst 的中位数
    data_lst 是一个 **排序后** 的 list
    data_lst的长度为: 4000
    data_lst[i] 长度为 784
    '''

    d = len(data_lst) / 2
    h, l = int(d), 0
    while l < h: 
        m = int((l + h) / 2)
        if data_lst[m][split_dim] < data_lst[h][split_dim]:
            l = m + 1
        else:
            h = m
    return data_lst[h], h


def getSplitDim(data_lst):
    """
    @:parameter
    data_lst: 不懂
    data_lst的长度为: 4000
    data_lst[i] 长度为 784

    计算points在每个维度上的和, 选择在和最大的维度上进行切割
    
    """
    
    # data_lst 相当于 4000*784 的 np.array, 在按行方向累加
    sum_lst = np.sum(data_lst, axis=0)  # 维度为 (784,)
    # 注意: A numpy array with shape (5,) is a 1 dimensional array while one with shape (5,1) is a 2 dimensional array.    
    split_dim = 0
    for v in range(1, len(sum_lst)):
        if sum_lst[v] > sum_lst[split_dim]:
            split_dim = v
    return split_dim


def buildKDTree(data_lst):
    #构建kd树
    # 确定在那个维度上进行分割
    split_dim = getSplitDim(data_lst) 
    data_lst = sorted(data_lst, key=lambda x: x[split_dim])
    # 选中值??
    point, m = findMedian(data_lst, split_dim)
    tree_node = KDNode(point, split_dim)

    if m > 0:
        tree_node.left = buildKDTree(data_lst[:m])
    if len(data_lst) > m + 1:
        tree_node.right = buildKDTree(data_lst[m + 1:])

    return tree_node


class NeiNode:
    '''neighbor node'''
    def __init__(self, p, d):
        self.__point = p
        self.__dist = d

    def get_point(self):
        return self.__point

    def get_dist(self):
        return self.__dist


class priorityQueue:
    '''优先队列'''
    def __init__(self, k):
        self.__K = k   # k近邻
        self.__pos = 0
        self.__priorityQueue = [0] * (k + 2)

    def add_neighbor(self, neighbor):
        self.__pos += 1
        self.__priorityQueue[self.__pos] = neighbor
        self.__swim_up(self.__pos)
        if self.__pos > self.__K:
            self.__exchange(1, self.__pos)
            self.__pos -= 1
            self.__sink_down(1)

    def get_knn_points(self):
        return [neighbor.get_point() for neighbor in self.__priorityQueue[1:self.__pos + 1]]

    def get_max_distance(self):
        if self.__pos > 0:
            return self.__priorityQueue[1].get_dist()
        return 0

    def get_knearest(self,k):
        if self.__pos > 0:
            tmp=[]
            while k > 0:
                tmp.append([self.__priorityQueue[k].get_dist(),self.__priorityQueue[k].get_point()])
                k = k-1
            return tmp
        return 0

    def is_full(self):
        return self.__pos >= self.__K

    def __swim_up(self, n):
        while n > 1 and self.__less(int(n / 2), n):
            self.__exchange(int(n / 2), n)
            n = n / 2

    def __sink_down(self, n):
        while 2 * n <= self.__pos:
            j = 2 * n
            if j < self.__pos and self.__less(j, j + 1):
                j += 1
            if not self.__less(n, j):
                break
            self.__exchange(n, j)
            n = j

    def __less(self, m, n):
        if m != 0:
            return self.__priorityQueue[m].get_dist() < self.__priorityQueue[n].get_dist()

    def __exchange(self, m, n):
        tmp = self.__priorityQueue[m]
        self.__priorityQueue[m] = self.__priorityQueue[n]
        self.__priorityQueue[n] = tmp

def knn_search_kd_tree_non_recursively(knn_priorityQueue, tree, target, search_track):
    track_node = []
    node_ptr = tree
    while node_ptr:
        while node_ptr:
            track_node.append(node_ptr)
            search_track.append([node_ptr.point, knn_priorityQueue.get_knn_points(), knn_priorityQueue.get_max_distance()])
            # 计算欧氏距离
            dist = np.linalg.norm(np.array(node_ptr.point) - np.array(target))

            knn_priorityQueue.add_neighbor(NeiNode(node_ptr.point, dist))

            search_track.append([None, knn_priorityQueue.get_knn_points(), knn_priorityQueue.get_max_distance()])

            split_dim = node_ptr.split_dim
            if target[split_dim] < node_ptr.point[split_dim]:
                node_ptr = node_ptr.left
            else:
                node_ptr = node_ptr.right

        while track_node:
            iter_node = track_node[-1]
            del track_node[-1]

            split_dim = iter_node.split_dim
            if not knn_priorityQueue.is_full() or \
                            abs(iter_node.point[split_dim] - target[split_dim]) < knn_priorityQueue.get_max_distance():
                if target[split_dim] < iter_node.point[split_dim]:
                    node_ptr = iter_node.right
                else:
                    node_ptr = iter_node.left

            if node_ptr:
                break
    a = knn_priorityQueue.get_knearest(k)
    return a


if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()


    time_1 = time.time()

    T = train_images
    nn = 3999
    mm = 399
    TT = [[0]*784]*(nn+1)
    TST = [[0]*784]*(mm+1)
    n = nn
    m = mm
    while n >= 0:
        TT[n] = list(itertools.chain.from_iterable(T[n]))
        n = n-1
    while m >= 0:
        TST[m] = list(itertools.chain.from_iterable(test_images[m]))
        m = m-1
    print('开始建树')
    kd_tree = buildKDTree(TT)
    print('建树结束\n')

    k = 2
    m = mm
    labelK = [0]*k
    labelResult = [0]*(m+1)
    j = 0
    print('begin to search target point in kd-tree')
    while j <= m:
        knn_priorityQueue = priorityQueue(k)
        search_track = []
        a = knn_search_kd_tree_non_recursively(knn_priorityQueue, kd_tree, TST[j], search_track)
        tmp1 = 0
        while tmp1 < k:
            labelK[tmp1] = TT.index(a[tmp1][1])
            tmp1 = tmp1+1
        #print(train_labels[labelK[:]])
        labelResult[j] = int(mode(train_labels[labelK[:]])[0][0])
        j = j+1
    lx = 0
    error = 0
    while lx <= m:
        if test_labels[lx] != labelResult[lx]:
            error = error+1
        lx = lx+1
    accuracy = 1-error/(m+1)
    print("accuracy: ", accuracy)

    time_2 = time.time()
    print('training cost ', time_2 - time_1,' second', '\n')
