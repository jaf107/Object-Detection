from typing import Union
from fastapi import FastAPI, UploadFile, HTTPException
import tensorflow_hub as hub
import tensorflow as tf
import time
import uvicorn

app = FastAPI()

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
    # You should implement this function to draw boxes on the image
    pass

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/get-version")
async def version():
    return {"1.0"}


# Define the endpoint for object detection
@app.post("/object_detection/")
async def object_detection(file: UploadFile):
    try:
        # Load the uploaded image
        img = load_img(file.file)
        
        # Convert and preprocess the image
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        
        # Perform object detection
        start_time = time.time()
        result = detector(converted_img)
        end_time = time.time()
        
        # Extract results
        result = {key: value.numpy() for key, value in result.items()}
        
        # Draw boxes on the image
        # image_with_boxes = draw_boxes(
        #     img.numpy(), result["detection_boxes"],
        #     result["detection_class_entities"], result["detection_scores"]
        # )
        
        # Return the results as JSON
        return {
            "objects_detected": len(result["detection_scores"]),
            "objects": result['detection_class_entities'],
            "inference_time": end_time - start_time
            # "image_with_boxes": image_with_boxes.tolist()  # Convert to list for JSON compatibility
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
