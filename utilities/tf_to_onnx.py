import tensorflow as tf 
import tf2onnx

input_path = "../data/temp/"
model_name = "saved_model"
file_type = ''
output_path = "../data/temp"

#pre_model = tf.keras.models.load_model(input_path + model_name + file_type)
#tf2onnx.convert.from_keras(
#    pre_model, 
#    output_path = output_path + model_name + ".onnx", 
#    opset = 9
#)

# maybe export model graph then convert from graph to onnx?
pre_model = tf.saved_model.load(input_path + model_name + file_type)
print(dir(pre_model))
print(pre_model.vars)
#tf2onnx.convert.from_tflite(
#    pre_model, 
#    output_path = output_path + model_name + ".onnx", 
#    opset = 9
#)