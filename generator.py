import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_real_samples
from load_data import make_fake_samples
from load_data import generate_real_samples

def define_generator(latent_dim):
    model = keras.models.Sequential()
    #4*4图片
    n_nodes = 256*3*3

    model.add(keras.layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    #调整输出形状（3，3，256）
    model.add(keras.layers.Reshape((3, 3, 256)))

    #上采样到8*8
    model.add(keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='valid'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))

    # 上采样到16*16
    model.add(keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))

    # 上采样到32*32
    model.add(keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))

    #output
    model.add(keras.layers.Conv2D(1, (3, 3), activation='tanh', padding='same'))

    return model

#生成latent_point作为生成器输入
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)  #正太分布[-1,1]
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(model_generator, latent_dim, n_samples):
    x = generate_latent_points(latent_dim,n_samples)

    xGen = model_generator.predict(x)
    yGen = np.zeros((n_samples,1))

    return xGen, yGen

def show_fake_sample():
    latent_dim = 100
    n_samples = 49

    generator = define_generator(latent_dim)

    xGen, yGen = generate_fake_samples(generator, latent_dim, n_samples)
    xGen = (xGen + 1) / 2.0  #[0,1]
    for i in range(n_samples):
        # define subplot
        plt.subplot(7, 7, 1 + i)
        # turn off axis labels
        plt.axis('off')
        # plot single image
        plt.imshow(xGen[i], cmap='gray_r')
    # show the figure
    plt.show()

if __name__ == '__main__':
    show_fake_sample()