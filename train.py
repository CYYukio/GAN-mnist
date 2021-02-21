import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from discriminator import *
from generator import *
from load_data import *
from GAN import *
def train(generator, discriminator, GAN, dataset, latent_dim, epochs=5, batch=128):
    per_epo = int(dataset.shape[0] / batch)  #每个epoch跑多少次batch
    half_batch = int(batch / 2)

    for i in range(epochs):
        for j in range(per_epo):
            ######训练判别器
            xReal, yReal = generate_real_samples(dataset, half_batch)
            discriminator_loss_real, _ = discriminator.train_on_batch(xReal, yReal)

            xFake, yFake = generate_fake_samples(generator, latent_dim, half_batch)
            discriminator_loss_fake, _ = discriminator.train_on_batch(xFake,yFake)
            ######

            ######训练生成器
            xGen = generate_latent_points(latent_dim, batch)
            yGen = np.ones((batch, 1))

            Gen_loss = GAN.train_on_batch(xGen, yGen)
            ######

            print('第>%d个epoch, %d/%d, 判别器real-loss：d1=%.3f, 判别器fake-loss：d2=%.3f Genloss：g=%.3f' %
                  (i + 1, j + 1, per_epo, discriminator_loss_real, discriminator_loss_fake, Gen_loss))

        # 每10次评估一下
        if (i + 1) % 10 == 0:
            summarize_performance(generator, discriminator, dataset, latent_dim)

    filename = 'minst_generator_model.h5'
    generator.save(filename)



def test_GAN():
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_GAN(g_model, d_model)
    # load image data
    dataset = load_real_samples()
    # train model
    train(g_model, d_model, gan_model, dataset, latent_dim)


if __name__ == '__main__':
    test_GAN()
