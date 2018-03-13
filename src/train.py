from keras.optimizers import Adam
import numpy as np

from discriminator import Discriminator
from generator import Generator
from gan import GAN
from config import load_data
from images import load_training_images

np.random.seed(4242)

# Parameters
optimizer = Adam(lr=0.0002, beta_1=0.5)
batch_size = 256
epochs = 100
random_dimension = 10
model_directory = './epochs'
loss_directory = './losses'
generated_images_directory = './generated'

# Load training data
images = load_training_images('mnist')

# Build GAN
generator = Generator(random_dimension)
discriminator = Discriminator()
gan = GAN(generator, discriminator, optimizer, random_dimension, model_directory, loss_directory, generated_images_directory)
gan.train(images, epochs, batch_size)
