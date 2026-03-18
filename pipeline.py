from ultralytics import YOLO
import cv2
from PIL import Image
from utils import risk_level

# Load YOLO model
yolo_model = YOLO("best.pt")


def run_pipeline(image_path):

    results = yolo_model(image_path, save=True)
    img = cv2.imread(image_path)

    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf

    diameters = []
    volumes = []
    confidences = []

    slice_thickness = 1

    # 🔍 Detection + feature extraction
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        width = x2 - x1
        height = y2 - y1

        diameter = max(width, height)
        area = width * height
        volume = area * slice_thickness

        diameters.append(diameter)
        volumes.append(volume)
        confidences.append(float(scores[i]))

    # 🧠 SIMPLE CLASSIFICATION (NO MODEL)
    if len(diameters) == 0:
        prediction = "normal"
        confidence_cls = 1.0
    else:
        if max(diameters) > 60:
            prediction = "malignant"
        else:
            prediction = "benign"

        confidence_cls = sum(confidences) / len(confidences)

    # ⚡ Risk
    risk = risk_level(len(boxes), diameters, prediction, confidences)

    # 🔥 Heatmap
    img_vis = cv2.imread(image_path)

    heatmap = cv2.applyColorMap(img_vis, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_vis, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("heatmap.jpg", heatmap)
    cv2.imwrite("overlay.jpg", overlay)

    return {
        "nodules": len(boxes),
        "max_diameter": max(diameters) if diameters else 0,
        "avg_volume": sum(volumes)/len(volumes) if volumes else 0,
        "prediction": prediction,
        "confidence": confidence_cls,
        "risk": risk,
        "image": image_path,
        # "heatmap_path": "overlay.jpg"
        "heatmap": heatmap,
        "overlay": overlay

    }
