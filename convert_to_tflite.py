import tensorflow as tf
import os

model_path = 'e:/masked_emotion/masked_emotion_cnn.h5'
tflite_path = 'e:/masked_emotion/masked_emotion_cnn.tflite'

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"Successfully saved TFLite model to {tflite_path}")
