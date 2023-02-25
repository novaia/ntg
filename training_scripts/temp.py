import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import tensorflow as tf

class TestModel(keras.Model):
    def __init__(self):
        super().__init__()

        #self.dense1 = layers.Dense(20, activation='relu')
        #self.dense2 = layers.Dense(10, activation='softmax')
        self.model = self.create_model()

    def create_model(self):
        inputs = layers.Input(shape=(784,))
        x = layers.Dense(20, activation='relu')(inputs)
        x = layers.Dense(10, activation='softmax')(x)
        return keras.Model(inputs, x)

    def call(self, inputs):
        #x = self.dense1(inputs)
        #x = self.dense2(x)
        x = self.model(inputs)
        return x

    def train_step(self, inputs):
        x, y = inputs

        with tf.GradientTape() as tape:
            probs = self(x)
            loss = self.compiled_loss(y, probs, regularization_losses=self.losses)
            gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return {'loss': loss}

(x_train, y_train), (_, _) = mnist.load_data()

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]).astype('float32') / 255
print('x_train new shape:', x_train.shape)

y_train = tf.one_hot(y_train, 10).numpy()
print('y_train new shape:', y_train.shape)

"""
SavedModel optimizer test.
"""
print('\nSavedModel optimizer test.\n')
# Train a model from scratch.
tm = TestModel()
tm.compile(optimizer='adam', loss='categorical_crossentropy')
tm.fit(x_train, y_train, epochs=10)

# Save the model.
tm.save('../data/test_model')

# Continue training.
tm = keras.models.load_model('../data/test_model')
tm.fit(x_train, y_train, epochs=10)