import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from discriminator import *
from generator import *
from load_data import *

def define_GAN(generator, discriminator):

    discriminator.trainable = False   #冻结判别器参数

    model = keras.models.Sequential()

    model.add(generator)
    model.add(discriminator)

    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def show_GAN():
    latent_dim = 100
    # create the discriminator
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator(latent_dim)
    # create the gan
    gan_model = define_GAN(generator, discriminator)
    # summarize gan model
    gan_model.summary()

def train_GAN(GAN, latent_dim, epochs=200, batch=128):
    for i in range(epochs):

        #GAN生成假的数据
        xGen = generate_latent_points(latent_dim, batch)
        yGen = np.ones((batch, 1))  #想要生成的假数据接近真的
        GAN.train_on_batch(xGen, yGen)
        #GAN的判别器冻住了，只会训练生成器，loss则是生成器生成的图“不像”真的


# 评估判别器acc，生成假图，保存模型
def summarize_performance(generator, discriminator, dataset, latent_dim, n_samples=150):

    #训练判别器
    #===================
    xReal, yReal = generate_real_samples(dataset, n_samples)
    _, acc_real = discriminator.evaluate(xReal, yReal, verbose=0)

    xFake, yFake = generate_fake_samples(generator, latent_dim, n_samples)
    _, acc_fake = discriminator.evaluate(xFake, yFake, verbose=0)
    print('判别器>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    #===================



