import matplotlib.pyplot as plt
import os
import numpy as np

from config import Configuration

def plot_loss(self, epoch, discriminator_losses, generator_losses):
    plt.figure(figsize=(10, 8))
    plt.plot(discriminator_losses, label='Discriminitive loss')
    plt.plot(generator_losses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('epochs/' + str(epoch) + '/gan_loss_epoch_%d.png' % epoch)


def plot_images(self, epoch, generator):
    examples=100
    dim=(10, 10)
    figsize=(10, 10)
    configuration = Configuration()
    noise = np.random.normal(0, 1, size=[examples, configuration.get_random_dim()])
    generated_images = generator.predict(noise)
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