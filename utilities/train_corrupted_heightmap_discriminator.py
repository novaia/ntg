from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Input, GlobalAveragePooling2D, Softmax
from keras.utils import image_dataset_from_directory 
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

batch_size = 32
image_size = 360
channels = 1

data_path = 'data/heightmap_discrimination_dataset'

def load_dataset(subset):
    dataset = image_dataset_from_directory(directory = data_path,
                                           label_mode = 'categorical',
                                           color_mode = 'grayscale',
                                           batch_size = batch_size,
                                           image_size = (image_size, image_size), 
                                           shuffle = True,
                                           validation_split = 0.2,
                                           subset = subset,
                                           seed = 10)
    return dataset

training = load_dataset('training')
validation = load_dataset('validation')

def create_model():
    model = Sequential()
    
    model.add(Input(shape=(image_size, image_size, channels)))
    
    model.add(Conv2D(128, kernel_size = (6, 6), strides = (4, 4), padding = 'same'))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(64, kernel_size = (3, 3)))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(32, kernel_size = (3, 3)))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(2, kernel_size = (3, 3)))
    model.add(ReLU())
    
    model.add(GlobalAveragePooling2D())
    model.add(Softmax())
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, 
                              min_delta = 0.001, mode = 'auto', verbose = 1)

model_checkpoint = ModelCheckpoint(monitor = 'val_accuracy', 
                                   save_best_only = True, 
                                   save_freq = 'epoch',
                                   filepath = 'data/corrupted_heightmap_discriminator.h5')

model = create_model()
model.fit(training, 
          validation_data = validation, 
          batch_size = batch_size, 
          epochs = 30, 
          callbacks = [reduce_lr, model_checkpoint])