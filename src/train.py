from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator
from config import Configuration
from gan import GAN

np.random.seed(4242)

from keras.datasets import mnist
(images, _), (_, _) = mnist.load_data()
images = (images.astype(np.float32) - 127.5)/127.5
images = images.reshape(60000, 784)

configuration = Configuration()
# images = configuration.load_data()
random_dim = configuration.get_random_dim()

optimizer = Adam(lr=0.0002, beta_1=0.5)
batch_size = 128
epochs = 50

# Build GAN
generator = Generator()
discriminator = Discriminator()
generator_network = generator.build(optimizer)
discriminator_network = discriminator.build(optimizer)

gan = GAN(generator_network, discriminator_network)
gan_network = gan.build(optimizer)

d_losses = []
g_losses = []

batch_count = int(images.shape[0] / batch_size)

def save_models(epoch):
    generator_network.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator_network.save('models/gan_discriminator_epoch_%d.h5' % epoch)

def plot_loss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(d_losses, label='Discriminitive loss')
    plt.plot(g_losses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)

def plot(epoch):
    examples=100
    dim=(10, 10)
    figsize=(10, 10)
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator_network.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch) 

for e in range(1, epochs + 1):    
    for _ in tqdm(range(batch_count)):
        noise = np.random.normal(0, 1, size=[batch_size, random_dim])
        image_batch = images[np.random.randint(0, images.shape[0], size=batch_size)]

        # Generate fake MNIST images
        generated_images = generator_network.predict(noise)
        X = np.concatenate([image_batch, generated_images])

        y_dis = np.zeros(2 * batch_size)
        y_dis[:batch_size] = 0.9

        # Train discriminator
        discriminator_network.trainable = True
        d_loss = discriminator_network.train_on_batch(X, y_dis)

        # Train generator
        noise = np.random.normal(0, 1, size=[batch_size, random_dim])
        y_gen = np.ones(batch_size)
        discriminator_network.trainable = False
        g_loss = gan_network.train_on_batch(noise, y_gen)

    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if e == 1 or e % 20 == 0:
        plot(e)
        save_models(e)

plot_loss(e)
