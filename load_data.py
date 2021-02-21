import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

def load_real_samples():
    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()
    X = trainX.reshape(trainX.shape[0], 28, 28, 1).astype('float32')

    X = X / 255

    return X


# 从dataset中生成n个real的数据+标签
def generate_real_samples(dataset, n_samples):
    # 生成0-数据集大小的n个随机数
    index = np.random.randint(0, dataset.shape[0], n_samples)
    #从dataset中取出
    xReal = dataset[index]
    #n行1列，real标签为1
    yReal = np.ones((n_samples, 1))

    return xReal, yReal

#生成假数据，用来对抗
def make_fake_samples(n_samples):
    #先生成28*28*1*n个数据【0，1】
    x = np.random.rand(28 * 28 * 1 * n_samples)
    xFake = x.reshape((n_samples, 28, 28, 1))               ##############
    yFake = np.zeros((n_samples, 1))

    return xFake, yFake
