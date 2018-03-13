from keras.datasets import cifar10
from keras.datasets import mnist
import numpy as np

from config import load_data

def load_training_images(type, **kwargs):
    if type is 'mnist':
        # MNIST dataset (use train and testing images to get more samples)
        (images, _), (test_images, _) = mnist.load_data()
        images = np.concatenate((images, test_images), axis=0)

    if type is 'cifar10':
        # CIFAR10 dataset
        class_to_train = int(kwargs['class_to_train'])
        (images, y_images), (test_images, y_test_images) = cifar10.load_data()
        images = images[np.array([item[0] for item in y_images]) == class_to_train]
        test_images = test_images[np.array([item[0] for item in y_test_images]) == 4]
        images = np.concatenate((images, test_images), axis=0)

    if type is 'local':
        # Local dataset
        images = load_data('./pictures')

    # Convert RGB values to float values
    images = (images.astype(np.float32) - 127.5) / 127.5
    # Resize images
    images = images.reshape((len(images), 28**2))
    return images