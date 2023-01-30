import tensorflow as tf 
import tf2onnx

input_path = "../data/models/diffusion_models/"
model_name = "diffusion1_epoch62"
file_type = ''
output_path = "../data/models/onnx_models/"

#load the model.
pre_model = tf.keras.models.load_model(input_path + model_name + file_type)

# Convert h5 to onnx.
tf2onnx.convert.from_keras(pre_model, output_path = output_path + model_name + ".onnx", opset = 9)