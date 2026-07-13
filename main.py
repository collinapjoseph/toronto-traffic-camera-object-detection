import os
import random
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from load_tc_data import get_toronto_traffic_cameras

# COCO class IDs that we are interested in.
VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}

# Color map for class bounding boxes
CLASS_COLORS = {
    "car":        "#0000ff",
    "truck":      "#ff8800",
    "bus":        "#ffff00",
    "motorcycle": "#00ff00",
    "bicycle":    "#ff00ff",
    "person":     "#ff0000",
}

# Load traffic camera info from Toronto Open Data API
TRAFFIC_CAMERAS = get_toronto_traffic_cameras()

# Directory to save labelled and unlabeled images
IMAGE_DIR = "_images"

# Options: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x 
# 'n' (nano/fastest) to 'x' (xlarge/most accurate)
MODEL_NAME = "yolo26m"

# Auto-downloads on first run
MODEL = YOLO(f"{MODEL_NAME}.pt")  

def load_tc_image(index):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    url = TRAFFIC_CAMERAS["IMAGEURL"][index]
    filename = url.split("/")[-1]
    dest = Path(IMAGE_DIR) / filename
    print(f"Downloading {filename}...")
    response = requests.get(url, timeout=10)
    dest.write_bytes(response.content)
    print(f"Saved to {Path(dest).resolve()}")
    return dest

def detect_vehicles(image_path, conf_threshold=0.35):
    results = MODEL.predict(
        source=image_path,
        conf=conf_threshold,
        verbose=False,
    )

    result = results[0]
    vehicle_boxes = []

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = MODEL.names[cls_id]

        if cls_name in VEHICLE_CLASS_NAMES:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            vehicle_boxes.append((x1, y1, x2, y2, conf, cls_name))

    return result, vehicle_boxes

def visualize_detections(image_path, vehicle_boxes, title=""):
    """
    Draw bounding boxes on the image and display it.
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    ax.axis("off")

    for (x1, y1, x2, y2, conf, cls_name) in vehicle_boxes:
        color = CLASS_COLORS.get(cls_name, "white")
        w, h = x2 - x1, y2 - y1

        rect = patches.FancyBboxPatch(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            boxstyle="round,pad=1",
        )
        ax.add_patch(rect)

        label = f"{cls_name} {conf:.0%}"
        ax.text(
            x1 + 4, y1 - 6, label,
            color="white",
            fontsize=9, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"),
        )

    n = len(vehicle_boxes)
    counts = {}
    for *_, cls_name in vehicle_boxes:
        counts[cls_name] = counts.get(cls_name, 0) + 1
    count_str = ", ".join(f"{v} {k}(s)" for k, v in counts.items())

    ax.set_title(
        f"{title} — {n} object(s) detected" + (f": {count_str}" if count_str else ""),
        fontsize=13, pad=10
    )
    plt.tight_layout()
    save_path = Path(IMAGE_DIR) / f"labelled_{Path(image_path).name}"
    plt.savefig(save_path)
    return save_path

def process_image(tcIndex):
    """
    (1) Load image from selected camera
    (2) Run inference on input image
    (3) Label and save image
    (4) Return file path to labelled image
    """
    image = load_tc_image(tcIndex)
    _, boxes = detect_vehicles(image, conf_threshold=0.35)
    imgPath = visualize_detections(image, boxes, title=Path(image).name)
    return imgPath

def main():
    print(f"Model  : {MODEL_NAME.upper()}")
    print(f"Task   : {MODEL.task}")
    print("Classes:")
    for cls_id, cls_name in sorted(MODEL.names.items()):
        if cls_name in VEHICLE_CLASS_NAMES:
            print(f"  [{cls_id}] {cls_name}")
    
    # Generate UI for inference.
    demo = gr.Interface(
        title="Toronto Traffic Camera Object Detection",
        fn=process_image,
        inputs=gr.Dropdown(
            type="index",
            choices=TRAFFIC_CAMERAS["INTERSECTION"].to_list(),
            label="Traffic Intersection"),
        outputs=gr.Image(type="filepath", label="Labelled Image")
    )
    demo.launch()

if __name__=="__main__":
    main()