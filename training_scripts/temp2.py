import keras
import tensorflow as tf

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]).astype('float32') / 255
y_train = tf.one_hot(y_train, 10).numpy()

model = keras.models.load_model('../data/test_model')
model.fit(x_train, y_train, epochs=10)