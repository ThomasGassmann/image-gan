from keras.optimizers import Adam
import numpy as np

from discriminator import Discriminator
from generator import Generator
from config import Configuration
from gan import GAN

np.random.seed(4242)

# Load test data
from keras.datasets import mnist
(images, _), (_, _) = mnist.load_data()
images = (images.astype(np.float32) - 127.5)/127.5
images = images.reshape(60000, 784)
# images = configuration.load_data()

# Parameters
optimizer = Adam(lr=0.0002, beta_1=0.5)
batch_size = 128
epochs = 50

# Build GAN
generator = Generator()
discriminator = Discriminator()
gan = GAN(generator, discriminator, optimizer)
gan.train(images, epochs, batch_size)
