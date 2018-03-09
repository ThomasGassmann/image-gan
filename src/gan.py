from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Optimizer
import os
import numpy as np
from tqdm import tqdm

from plot import plot_loss, plot_images
from discriminator import Discriminator
from generator import Generator

class GAN:
    def __init__(self, generator: Generator, discriminator: Discriminator, optimizer: Optimizer, random_dimension):
        self.random_dim = random_dimension
        self.generator = generator.build(optimizer)
        self.discriminator = discriminator.build(optimizer)
        self.discriminator_losses = []
        self.generator_losses = []
        self.optimizer = optimizer


    def save_models(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        generator_path = os.path.join(directory, 'generator.h5')
        discriminator_path = os.path.join(directory, 'discriminator.h5')
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)


    def train(self, images, epochs, batch_size):
        gan = self.__build()
        batch_count = int(images.shape[0] / batch_size)
        for e in range(1, epochs + 1):    
            for _ in tqdm(range(batch_count)):
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                image_batch = images[np.random.randint(0, images.shape[0], size=batch_size)]

                # Generate fake MNIST images
                generated_images = self.generator.predict(noise)
                # Combine generated generated images and training images
                X = np.concatenate([image_batch, generated_images])

                # Labels for training and generated images
                y_dis = np.zeros(2 * batch_size)
                # 0 is fake, 1 is real
                # Label smoothing (see: https://github.com/aleju/papers/blob/master/neural-nets/Improved_Techniques_for_Training_GANs.md)
                # All images from 0 to batch_size are (caused by concatenation) real
                y_dis[:batch_size] = 0.9
                y_dis[batch_size:] = 0.1

                # Train discriminator
                d_loss = self.discriminator.train_on_batch(X, y_dis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                y_gen = np.ones(batch_size)
                g_loss = gan.train_on_batch(noise, y_gen)

            self.discriminator_losses.append(d_loss)
            self.generator_losses.append(g_loss)

            plot_images(e, self.generator, self.random_dim)
            self.save_models('./epochs/' + str(e))

        plot_loss(e, self.discriminator_losses, self.generator_losses)


    def __build(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.random_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return gan
