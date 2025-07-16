# app.py (Flask application for model serving)

from flask import Flask, request, jsonify
import onnxruntime as rt
from PIL import Image
import io
import base64
import numpy as np
from torchvision import transforms # Used for image preprocessing

app = Flask(__name__)

# --- Model Loading ---
# Path to your ONNX model file (it will be in the same directory as app.py in the Docker container)
ONNX_MODEL_PATH = "mammogram_model34.onnx" # Updated ONNX model name

# Load the ONNX model when the application starts
try:
    sess = rt.InferenceSession(ONNX_MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX model loaded successfully from {ONNX_MODEL_PATH}")
    print(f"Model Input Name: {input_name}, Output Name: {output_name}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    # Note: The original message said "mammogram_model.onnx" but you changed it to "mammogram_model34.onnx"
    # Ensure the file name matches exactly what you have.
    print(f"Please ensure '{ONNX_MODEL_PATH}' is in the same directory as app.py.")
    exit(1) # Exit if model cannot be loaded, as the service won't function

# --- Image Preprocessing Transformations ---
# These transformations MUST exactly match the 'val' transformations used during your model training.
# Ensure the mean and std values are identical to what you used in your PyTorch training.
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Define your class names consistently with your training ---
FINAL_CLASS_NAMES = ['NORMAL', 'ABNORMAL'] # Ensure this matches your model's output classes

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Receive image data from the request
    # Expecting a JSON body like: {"image": "base64_encoded_image_string"}
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data provided in JSON body. Expected {"image": "base64_string"}'}), 400

    try:
        image_data_base64 = request.json['image']
        # Decode the Base64 string back to bytes
        image_bytes = base64.b64decode(image_data_base64)
        # Open the image using PIL (Pillow)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        # Log the error for debugging
        print(f"Error decoding or opening image: {e}")
        return jsonify({'error': f'Invalid image data or Base64 decoding error: {e}'}), 400

    # 2. Preprocess the image for model input
    try:
        # Apply the defined transformations
        input_tensor = inference_transform(image)
        # Add a batch dimension (model expects [batch_size, channels, height, width])
        input_tensor = input_tensor.unsqueeze(0)
        # Convert to NumPy array, as ONNX Runtime expects NumPy inputs
        input_numpy = input_tensor.numpy()
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return jsonify({'error': f'Image preprocessing failed: {e}'}), 500

    # 3. Perform inference using the ONNX model
    try:
        # Run inference using the ONNX Runtime session
        # The output is a list, so we take the first element (which is the actual output array)
        output_logits = sess.run([output_name], {input_name: input_numpy})[0]
        
        # Apply softmax to get probabilities if your model outputs logits (common for classification)
        # np.exp(x) / np.sum(np.exp(x)) is the softmax function
        probabilities = np.exp(output_logits) / np.sum(np.exp(output_logits), axis=1, keepdims=True)
        
        # Get the index of the class with the highest probability
        predicted_class_idx = np.argmax(probabilities, axis=1)[0]
        
        # Map the predicted index to the human-readable class name
        prediction_label = FINAL_CLASS_NAMES[predicted_class_idx]
        
        # Convert probabilities to a standard Python list for JSON serialization
        probabilities_list = probabilities.tolist()[0]

    except Exception as e:
        print(f"Error during model inference: {e}")
        return jsonify({'error': f'Model inference failed: {e}'}), 500

    # 4. Return the prediction as a JSON response
    return jsonify({
        'prediction': prediction_label,
        'probabilities': probabilities_list
    })

# --- Run the Flask application ---
if __name__ == '__main__':
    # When deploying with Docker or a service, host='0.0.0.0' is important
    # to make the server accessible from outside the container.
    # The port 5000 is chosen as a common port for Flask apps.
    app.run(host='0.0.0.0', port=5000)
