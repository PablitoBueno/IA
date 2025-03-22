# Install required libraries
!pip install fastapi uvicorn transformers torch opencv-python numpy pillow

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoProcessor, AutoModelForDepthEstimation
import torch
import cv2
import numpy as np
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Load the processor and model
model_id = "Intel/zoedepth-nyu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id)

def preprocess_image(image_bytes: bytes):
    """Loads and preprocesses the image from bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    return img, inputs

def generate_depth_map(inputs):
    """Generates the depth map from the preprocessed image."""
    with torch.no_grad():
        outputs = model(**inputs)
    depth_map = outputs.predicted_depth[0].squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)
    return depth_map

def apply_smoothing(depth_map, kernel_size: int = 5):
    """Applies GaussianBlur smoothing to the depth map."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), 0)

def apply_threshold(depth_map, min_depth: int, max_depth: int):
    """Applies thresholding to highlight objects within a depth range."""
    _, thresholded = cv2.threshold(depth_map, min_depth, max_depth, cv2.THRESH_BINARY)
    return thresholded

def visualize_depth_map(depth_map, mode: str = "gray", original_img: np.ndarray = None):
    """Converts the depth map into the requested visualization mode."""
    if mode == "gray":
        return depth_map
    elif mode == "color":
        return cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)
    elif mode == "overlay" and original_img is not None:
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        return cv2.addWeighted(original_img, 0.6, depth_colored, 0.4, 0)
    else:
        raise ValueError("Invalid mode or original image not provided for overlay.")

def detect_edges(depth_map):
    """Detects edges in the depth map using the Canny algorithm."""
    edges = cv2.Canny(depth_map, 50, 150)
    return edges

@app.post("/process_depth_map")
async def process_depth_map(
    file: UploadFile = File(..., description="Image to be processed"),
    mode: str = Query("gray", description="Visualization mode: 'gray', 'color', or 'overlay'"),
    smoothing: bool = Query(False, description="Apply smoothing?"),
    kernel_size: int = Query(5, description="Kernel size for smoothing (odd number)"),
    threshold: bool = Query(False, description="Apply threshold?"),
    min_depth: int = Query(50, description="Minimum depth value for threshold"),
    max_depth: int = Query(200, description="Maximum depth value for threshold"),
    edges: bool = Query(False, description="Apply edge detection?")
):
    # Read and preprocess the image
    image_bytes = await file.read()
    try:
        original_img, inputs = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing the image.")

    # Generate depth map
    depth_map = generate_depth_map(inputs)

    # Apply smoothing if requested
    if smoothing:
        depth_map = apply_smoothing(depth_map, kernel_size=kernel_size)

    # Apply threshold if requested
    if threshold:
        depth_map = apply_threshold(depth_map, min_depth, max_depth)

    # Apply edge detection if requested
    if edges:
        depth_map = detect_edges(depth_map)

    # Configure visualization
    if mode == "overlay":
        original_img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        try:
            result_img = visualize_depth_map(depth_map, mode=mode, original_img=original_img_cv)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        result_img = visualize_depth_map(depth_map, mode=mode)

    # Convert result to bytes (PNG)
    is_success, buffer = cv2.imencode(".png", result_img)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to generate image.")
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
