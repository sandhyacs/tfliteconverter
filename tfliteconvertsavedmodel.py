import tensorflow as tf

# Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model("./saved_model_updated1") # path to the SavedModel directory
# tflite_model = converter.convert()

# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)

# import tensorflow as tf

# saved_model_dir = '/pretrained_model/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8/saved_model'
saved_model_dir = './saved_model'

# model = tf.saved_model.load(saved_model_dir)
# model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[0].set_shape([1, 640, 640, 3])
# tf.saved_model.save(model, "saved_model_updated", signatures=model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY])
# # Convert
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir='saved_model_updated', signature_keys=['serving_default'])
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)

# # ## TFLite Interpreter to check input shape
# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Test the model on random input data.
# input_shape = input_details[0]['shape']
# print(input_shape)

# import tensorflow as tf
# converter = tf.lite.TFLiteConverter.from_saved_model("./saved_model")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quant_model = converter.convert()


model = tf.saved_model.load(saved_model_dir)
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, 1280, 720, 3])
converter = TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open('modelconc.tflite', 'wb') as f:
  f.write(tflite_model)
