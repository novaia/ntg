import tensorflow as tf
from keras import models
from train_diffusion import reverse_diffusion, save_images

starting_epoch = 2
ending_epoch = 46
model_path = '../data/models/diffusion_models/'
model_name = 'diffusion1'
image_save_path = '../data/images/'

if __name__ == '__main__':
    for epoch in range(starting_epoch, ending_epoch):
        model = models.load_model(model_path + model_name + '_epoch_' + str(epoch))

        initial_noise = tf.random.normal(shape=(7, 256, 256, 1))

        generated_images = reverse_diffusion(model, num_images = 7, diffusion_steps = 20, initial_noise = initial_noise)
        save_images(generated_images, rows = 1, columns = 7, epoch = epoch)