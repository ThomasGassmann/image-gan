from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers

from config import Configuration

class Generator:
    def __init__(self):
        configuration = Configuration()
        self.random_dim = configuration.get_random_dim()

    def build(self, optimizer):
        generator = Sequential()
        generator.add(Dense(256, input_dim=self.random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(784, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return generator