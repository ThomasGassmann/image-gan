from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np

from discriminator import Discriminator
from generator import Generator
from gan import GAN
from config import load_data

np.random.seed(4242)

# MNIST dataset (use train and testing images to get more samples)
(images, _), (test_images, _) = mnist.load_data()
images = np.concatenate((images, test_images), axis=0)

# Local dataset
images = load_data('./pictures')

# Convert RGB values to float values
images = (images.astype(np.float32) - 127.5) / 127.5
# Resize images
images = images.reshape(len(images), 784)

# Parameters
optimizer = Adam(lr=0.0002, beta_1=0.5)
batch_size = 10
epochs = 1000
random_dimension = 10

# Build GAN
generator = Generator(random_dimension)
discriminator = Discriminator()
gan = GAN(generator, discriminator, optimizer, random_dimension)
gan.train(images, epochs, batch_size)
