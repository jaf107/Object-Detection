from typing import Union
from fastapi import FastAPI, UploadFile, File
import tensorflow_hub as hub
import tensorflow as tf
import time
import cv2
import uvicorn
import numpy as np

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
    # Convert image to NumPy array
    image_np = image.numpy()
    
    for i in range(len(boxes)):
        box = boxes[i]
        class_entity = class_entities[i].decode('utf-8')
        score = scores[i]
        
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = int(xmin * image_np.shape[1]), int(xmax * image_np.shape[1]), \
                                   int(ymin * image_np.shape[0]), int(ymax * image_np.shape[0])
        
        # Draw bounding box on the image
        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_np, f"{class_entity} ({score:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_np

@app.route("/getversion")
def version():
    return jsonify({"version": "1.0"})

@app.get("/getversion")
async def version():
    return {"1.0"}

# Define the endpoint for object detection
@app.post("/object_detection")
async def object_detection(file: UploadFile=File(...)):
    print(file.filename)
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
        img = load_img(file.file)
        print("hello")
        
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
            img, result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"]
        )
        print("after draw booxes")
        
        # Return the results as JSON
        return {
            "objects": result['detection_class_entities'].tolist()
        }
    except Exception as e:
        return {"error": str(e)}

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
