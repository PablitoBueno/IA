# Install required libraries
!pip install fastapi uvicorn opencv-python numpy base64

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI()

# CORS configuration to allow front-end (React) requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000"],  # Allows front-end to access the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_yolo_model(weights_path="yolov3.weights", config_path="yolov3.cfg"):
    """
    Loads the YOLO network from weights and configuration files.
    """
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

def load_classes(names_path="coco.names"):
    """
    Loads class names from a file.
    """
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def process_image(img, net, output_layers, classes, confidence_threshold=0.5):
    """
    Processes the image using the YOLO model and returns detections and the annotated image.
    """
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    detections = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                detections.append({
                    "class": classes[class_id],
                    "confidence": float(confidence),
                    "box": [x, y, w, h]
                })

                # Draw bounding box and label on the image
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{classes[class_id]} {confidence:.2f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detections, img

# Load model and classes once at API startup
net, output_layers = load_yolo_model("yolov3.weights", "yolov3.cfg")
classes = load_classes("coco.names")

@app.post("/detect/")
async def detect(image: UploadFile = File(...)):
    """
    Endpoint that receives an image via POST and returns detection data and the annotated image.
    """
    file_bytes = await image.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image!"}

    detections, annotated_img = process_image(img, net, output_layers, classes)

    # Encode the annotated image in base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    return {
        "detections": detections,
        "annotated_image": jpg_as_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
