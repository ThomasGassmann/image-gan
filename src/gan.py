from keras.layers import Input
from keras.models import Model

from discriminator import Discriminator
from generator import Generator
from config import Configuration

class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.configuration = Configuration()

    def build(self, optimizer):
        self.discriminator.trainable = False
        random_dim = self.configuration.get_random_dim()
        gan_input = Input(shape=(random_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
        return gan
