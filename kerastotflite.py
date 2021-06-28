import tensorflow as tf
import sys
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import CustomObjectScope

tf.enable_resource_variables()

def relu6(x):
	return K.relu(x, max_value=6)

with CustomObjectScope({'relu6': relu6}):
	tflite_model = tf.lite.TFLiteConverter.from_keras_model_file(sys.argv[1]).convert()
	with open(sys.argv[2], 'wb') as f:
		f.write(tflite_model)
