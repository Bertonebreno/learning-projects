import cv2
import pandas as pd

from yolo.display import save_detections
from yolo.engine import apply_nms, forward, get_confident_boxes

images = [
    "cat.jpg",
    "city_scene.jpg",
    "dog.jpg",
    "dog2.jpg",
    "eagle.jpg",
    "food.jpg",
    "garfo_meio_tampado.jpeg",
    "garfo_tampado.jpeg",
    "giraffe.jpg",
    "horses.jpg",
    "mesa.jpeg",
    "mesa_cadeira.jpeg",
    "motorbike.jpg",
    "person.jpg",
    "surf.jpg",
    "talheres.jpeg",
    "wine.jpg",
]

configs_to_try = [
    {"min_confidence": 0.5, "iou_threshold": 0.4},
    {"min_confidence": 0.5, "iou_threshold": 0.75},
    {"min_confidence": 0.4, "iou_threshold": 0.4},
    {"min_confidence": 0.75, "iou_threshold": 0.4},
]

info = []

for img in images:
    for config in configs_to_try:
        image_name = img.split(".")[0]
        print(f"Processing {image_name} with {config}")

        image_path = f"images/{img}"
        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        output = forward(image)
        class_ids, scores, boxes = get_confident_boxes(
            output, height, width, min_confidence=config["min_confidence"]
        )
        chosen_boxes = apply_nms(boxes, scores, iou_threshold=config["iou_threshold"])

        save_detections(
            image,
            f"{image_name}-{config["min_confidence"]}-{config["iou_threshold"]}",
            boxes,
            class_ids,
            scores,
            chosen_boxes,
        )

        info.append(
            {
                "image_name": image_name,
                "min_confidence": config["min_confidence"],
                "iou_threshold": config["iou_threshold"],
                "number_of_detections": len(chosen_boxes),
            }
        )

pd.DataFrame(info).sort_values(by="image_name").to_csv("results/info.csv")
