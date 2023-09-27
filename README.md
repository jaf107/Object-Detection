# Object-Detection


This Git repository contains code for an Object Detection API built using FastAPI. The main functionality of this API is to perform object detection on images using a pre-trained model and provide the results in a user-friendly format.

## Usage

To run the Object Detection API, make sure you have FastAPI and Uvicorn installed. If not, you can install them using `pip`:

```bash
pip install -r requirements.txt
```

Once you have FastAPI and Uvicorn installed, you can start the API with the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

This will start the FastAPI server, and you should see output indicating that the server is running.

### Endpoints

The API provides the following endpoint:

- `POST /detect`: Upload an image for object detection. The API will return the detected objects in the image along with their bounding boxes.

To test the API, you can use a tool like `curl` or write your own code to interact with it programmatically. Here's an example of how to use `curl` to test the API:

```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:8000/detect
```

Replace `"path/to/your/image.jpg"` with the actual path to the image you want to use for testing.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or suggestions, please feel free to contact us at [Jafar](mailto:bsse1109@iit.du.ac.bd).

Happy coding! 🚀
