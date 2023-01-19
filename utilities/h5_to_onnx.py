import tensorflow as tf 
import tf2onnx

save_path = "data/"
model_name = "diffusion1_epoch_200"

#load the model.
pre_model = tf.keras.models.load_model(save_path + model_name + ".h5")

# Convert h5 to onnx.
tf2onnx.convert.from_keras(pre_model, output_path = save_path + model_name + ".onnx")