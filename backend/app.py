import os
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
from flask_cors import CORS
try:
    # pyrefly: ignore [missing-import]
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback to tf.lite if someone runs this locally with full TF installed
    # pyrefly: ignore [missing-import]
    import tensorflow.lite as tflite
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

app = Flask(__name__)
# Enable CORS so the React app can talk to this backend
CORS(app)

# Load the TFLITE model relative to the backend directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'masked_emotion_cnn.tflite')
interpreter = None
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Successfully loaded TFLite model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'error': 'Model failed to load on the server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    try:
        # Read image bytes
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode the image as grayscale
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
            
        # Preprocess as required by the model (48x48, grayscale, normalized)
        img_resized = cv2.resize(img, (48, 48))
        img_reshaped = np.reshape(img_resized, (1, 48, 48, 1)).astype(np.float32) / 255.0
        
        # Predict using TFLite
        interpreter.set_tensor(input_details[0]['index'], img_reshaped)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        
        return jsonify({
            'emotion': EMOTIONS[predicted_class],
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)
