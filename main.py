from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow_hub as hub
import tensorflow as tf
import time
import cv2
import numpy as np
import os

app = Flask(__name__)

# Define the model module handle
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

# Load the model and get the detector function
detector = hub.load(module_handle).signatures['default']

# Define a function to load an image
def load_img(file):
    img = tf.image.decode_image(file.read(), channels=3)
    return img

# Define a function to draw boxes on the image
def draw_boxes(image, boxes, class_entities, scores):
    for box, class_entity, score in zip(boxes[0], class_entities[0], scores[0]):
        ymin, xmin, ymax, xmax = box
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{class_entity.decode('utf-8')} ({int(100 * score)}%)"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

@app.route("/getversion")
def version():
    return jsonify({"version": "1.0"})

@app.route("/object_detection/", methods=["POST"])
def object_detection():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        # Check if the file has a valid filename
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        # Check if the file extension is allowed (you can extend this list)
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file extension"})

        # Load the uploaded image
        img = load_img(file)

        # Convert and preprocess the image
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

        # Perform object detection
        start_time = time.time()
        result = detector(converted_img)
        end_time = time.time()

        # Extract results
        result = {key: value.numpy() for key, value in result.items()}

        # Draw boxes on the image
        image_with_boxes = draw_boxes(
            img.numpy(), result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"]
        )

        # Convert the image with boxes to bytes
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        image_with_boxes_bytes = img_encoded.tobytes()

        # Return the results as JSON
        return jsonify({
            "objects_detected": len(result["detection_scores"]),
            "objects": result['detection_class_entities'].tolist(),
            "inference_time": end_time - start_time,
            "image_with_boxes": image_with_boxes_bytes  # Convert to bytes for image response
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
