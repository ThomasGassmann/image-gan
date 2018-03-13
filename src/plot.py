import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.misc
from PIL import Image
import uuid

def plot_loss(epoch, discriminator_losses, generator_losses, directory):
    create_dir_if_not_exists(directory)
    plt.figure(figsize=(10, 10))
    plt.plot(discriminator_losses, label='Discriminitive loss')
    plt.plot(generator_losses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    file_name = os.path.join(directory, f'loss-e{epoch}.png')
    plt.savefig(file_name)


def plot_images(epoch, generator, random_dim, directory):
    create_dir_if_not_exists(directory)
    examples=100
    figsize=(10, 10)
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(figsize[0], figsize[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    file_name = os.path.join(directory, f'generated_images_e-{epoch}.png')
    plt.savefig(file_name)


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)