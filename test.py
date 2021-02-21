import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from discriminator import *
from generator import *
from load_data import *

# plot the generated images
def create_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :], cmap='gray')
    plt.show()

def show_imgs_for_final_generator_model():
    # load model
    model = tf.keras.models.load_model('minst_generator.h5')
    # generate images
    latent_points = generate_latent_points(100, 100)
    # generate images
    X = model.predict(latent_points)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot the result
    X = X.reshape(X.shape[0], 28,28)
    create_plot(X, 10)

if __name__ == '__main__':
    show_imgs_for_final_generator_model()