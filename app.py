import os
import json
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Load the model and class names
model = load_model("best_model.h5")

# Load class names from the JSON file
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Initialize Flask app
app = Flask(__name__)

# Function to compute optical flow (same as in the previous code)
def compute_optical_flow(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    flows = []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while len(flows) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.resize(flow, (112, 112))
        flows.append(flow)
        prev_gray = gray

    cap.release()

    flows = flows + [flows[-1]] * (num_frames - len(flows))
    flows = flows[:num_frames]

    flows = np.array(flows, dtype=np.float32)
    return flows

# Function to preprocess keypoints for model input
def preprocess_keypoints(keypoints, num_frames=16):
    return keypoints.reshape(1, num_frames, -1)  # Flatten the flow to the model's expected input

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    # Save the uploaded video file
    video_file = request.files['video']
    video_path = os.path.join(video_file.filename)
    video_file.save(video_path)

    # Extract optical flow features
    keypoints = compute_optical_flow(video_path)
    if keypoints is None:
        os.remove(video_path)
        return jsonify({"error": "Failed to extract optical flow from video"}), 400

    # Preprocess the optical flow and make prediction
    keypoints = preprocess_keypoints(keypoints)
    prediction = model.predict(keypoints)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]  # Get class name from index

    # Clean up the temporary video file
    os.remove(video_path)

    # Return the class name
    return jsonify({"class_name": class_name})

# Define the status endpoint
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"success": True, "code": 200})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)