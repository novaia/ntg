from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import layers
from keras import models
from keras import activations
from keras.losses import mean_absolute_error
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

starting_epoch = 47 # 0 if training from scratch.
data_path = '../../heightmaps/uncorrupted_split_heightmaps_second_pass/'
model_save_path = '../data/models/diffusion_models/'
model_name = 'diffusion1'
image_save_path = '../data/images/'

# Sampling.
min_signal_rate = 0.02
max_signal_rate = 0.95

# Architecture.
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# Optimization.
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = Adam(0.0002, 0.5)

# Input.
batch_size = 8
image_size = 256
channels = 1

def main():
    print(tf.__version__)

    print('1: ', tf.config.list_physical_devices('GPU'))
    print('2: ', tf.test.is_built_with_cuda)
    print('3: ', tf.test.gpu_device_name())
    print('4: ', tf.config.get_visible_devices())

    idg = ImageDataGenerator(preprocessing_function = preprocessing_function)
    heightmap_iterator = idg.flow_from_directory(data_path, 
                                                 target_size = (image_size, image_size), 
                                                 batch_size = batch_size,
                                                 color_mode = 'grayscale',
                                                 classes = [''])

    if starting_epoch == 0:
        model = create_model(image_size, widths, block_depth)
        model.compile(optimizer=optimizer, loss=mean_absolute_error)
    else:
        model = models.load_model(model_save_path + model_name + '_epoch' + str(starting_epoch))

    model = train(heightmap_iterator, model, 4)


def preprocessing_function(image):
    image = image.astype(float) / 255
    return image


def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(tf.linspace(tf.math.log(embedding_min_frequency),
                                     tf.math.log(embedding_max_frequency),
                                     embedding_dims // 2))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def create_model(image_size, widths, block_depth):
    noisy_images = layers.Input(shape=(image_size, image_size, channels))
    noise_variances = layers.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(channels, kernel_size=1, kernel_initializer="zeros")(x)

    return models.Model([noisy_images, noise_variances], x, name="residual_unet")


def diffusion_schedule(diffusion_times):
    # diffusion times -> angles
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    # angles -> signal and noise rates
    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates


def reverse_diffusion(model, num_images, diffusion_steps, initial_noise = None):
    if initial_noise == None:
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, channels))
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        
        pred_noises = model([noisy_images, noise_rates**2], training = True)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(next_diffusion_times)
        next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
        
    return pred_images


def save_images(images, rows, columns, epoch):
    f, axs = plt.subplots(rows, columns, figsize=(100,35))
    i = 0
    for x in axs.flatten():
        x.imshow(images[i], cmap='gray')
        x.axis('off')
        i += 1
    plt.tight_layout()
    plt.savefig(image_save_path + model_name + '_epoch' + str(epoch) + '.png')


def train(directory_iterator, model, epochs):
    directory_iterator.reset()
    steps_per_epoch = directory_iterator.__len__() - 1
    
    for epoch in range(epochs):
        epoch_start_time = datetime.now()

        for step in range(steps_per_epoch):
            images = np.asarray(directory_iterator.next()[0])
            if images.shape[0] != batch_size:
                continue
                
            noises = tf.random.normal(shape=(batch_size, image_size, image_size, channels))

            diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
            noise_rates, signal_rates = diffusion_schedule(diffusion_times)

            noisy_images = signal_rates * images + noise_rates * noises
            model.train_on_batch([noisy_images, noise_rates**2], noises)
        
        epoch_end_time = datetime.now()
        epoch_delta_time = epoch_end_time - epoch_start_time
        simple_epoch_end_time = str(epoch_end_time.hour) + ':' + str(epoch_end_time.minute)
        
        absolute_epoch = starting_epoch + epoch + 1
        
        print('Epoch ' + str(absolute_epoch) + ' completed at ' + str(simple_epoch_end_time) + ' in ' + str(epoch_delta_time))
        model.save(model_save_path + model_name + '_epoch' + str(absolute_epoch))

        generated_images = reverse_diffusion(model, num_images = 7, diffusion_steps = 20)
        save_images(generated_images, rows = 1, columns = 7, epoch = absolute_epoch)
    
    return model


if __name__ == '__main__':
    main()
