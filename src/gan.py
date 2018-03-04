from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Optimizer
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator
from config import Configuration

class GAN:
    def __init__(self, generator: Generator, discriminator: Discriminator, optimizer: Optimizer):
        self.generator = generator.build(optimizer)
        self.discriminator = discriminator.build(optimizer)
        self.configuration = Configuration()
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
        self.__build()
        random_dim = self.configuration.get_random_dim()
        batch_count = int(images.shape[0] / batch_size)
        for e in range(1, epochs + 1):    
            for _ in tqdm(range(batch_count)):
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
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

                # Train discriminator
                d_loss = self.discriminator.train_on_batch(X, y_dis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                y_gen = np.ones(batch_size)
                g_loss = self.gan.train_on_batch(noise, y_gen)

            self.discriminator_losses.append(d_loss)
            self.generator_losses.append(g_loss)

            if e == 1 or e % 20 == 0:
                self.__plot(e)
                self.save_models('./epochs/' + str(e))

        self.__plot_loss(e)


    def __build(self):
        self.discriminator.trainable = False
        random_dim = self.configuration.get_random_dim()
        gan_input = Input(shape=(random_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.gan = gan


    def __plot_loss(self, epoch):
        plt.figure(figsize=(10, 8))
        plt.plot(self.discriminator_losses, label='Discriminitive loss')
        plt.plot(self.generator_losses, label='Generative loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('epochs/' + str(epoch) + '/gan_loss_epoch_%d.png' % epoch)


    def __plot(self, epoch):
        examples=100
        dim=(10, 10)
        figsize=(10, 10)
        noise = np.random.normal(0, 1, size=[examples, self.configuration.get_random_dim()])
        generated_images = self.generator.predict(noise)
        generated_images = generated_images.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        if not os.path.exists('images'):
            os.makedirs('images')
        plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)
