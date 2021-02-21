import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_real_samples
from load_data import make_fake_samples
from load_data import generate_real_samples
# 实现discriminator
def define_discriminator(in_shape=(28, 28, 1)):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=in_shape))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))

    # 分类
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.4))  # 提高泛化
    model.add(keras.layers.Dense(units=1, activation="sigmoid"))

    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model


#训练判别器
def train_discriminator(model_discriminator, dataset, n_itr=20, n_batch=128):
    #训练判别器，迭代数20，batch数128，半真半假，各64

    half_batch = int(n_batch / 2)

    for i in range(n_itr):
        xReal, yReal = generate_real_samples(dataset, half_batch)  #抽取真数据

        _, real_acc = model_discriminator.train_on_batch(xReal, yReal)

        xFake, yFake = make_fake_samples(half_batch)   #生成假数据

        _, fake_acc = model_discriminator.train_on_batch(xFake, yFake)

        print('判别器准确率： 第>%d real=%.0f%% fake=%.0f%%' % (i + 1, real_acc * 100, fake_acc * 100))

def test_discriminator():
    discriminator = define_discriminator()
    dataset = load_real_samples()

    train_discriminator(discriminator, dataset)


if __name__ == '__main__':
    test_discriminator()