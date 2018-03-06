import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.misc
from PIL import Image
import uuid

def plot_loss(epoch, discriminator_losses, generator_losses):
    plt.figure(figsize=(10, 8))
    plt.plot(discriminator_losses, label='Discriminitive loss')
    plt.plot(generator_losses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('epochs/' + str(epoch) + '/gan_loss_epoch_%d.png' % epoch)


def plot_images(epoch, generator, random_dim):
    examples=100
    dim=(10, 10)
    figsize=(10, 10)
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 256, 256)
    if not os.path.exists('./images/examples'):
        os.makedirs('./images/examples')

    plt.figure(figsize=figsize)
    guid = uuid.uuid4()
    plt.imshow(generated_images[0], interpolation='nearest', cmap='gray_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/' + str(guid) + '.png')