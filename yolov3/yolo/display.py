from typing import List, Optional, Tuple

import cv2
import numpy as np

with open("files/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def draw_box(
    image: np.ndarray,
    box: List[int],
    label: Optional[str] = None,
    confidence: Optional[float] = None,
) -> None:
    x, y, w, h = box[0], box[1], box[2], box[3]

    color = red_to_green_gradient(confidence)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 8)

    if label is not None:
        cv2.putText(
            image,
            f"{label} {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def red_to_green_gradient(
    value: Optional[float], power: int = 3
) -> Tuple[int, int, int]:
    if value is None:
        return (0, 0, 0)

    if value < 0 or value > 1:
        raise ValueError("Input must be between 0 and 1")

    adjusted_value = value**power

    red = int(255 * (1 - adjusted_value))
    green = int(255 * adjusted_value)
    blue = 0

    return (blue, green, red)


def show_detections(
    image: np.ndarray,
    boxes: List[List[int]],
    class_ids: List[int],
    scores: List[float],
    chosen_boxes: List[int],
) -> None:
    for i in chosen_boxes:
        box = boxes[i]

        label = str(classes[class_ids[i]])
        confidence = scores[i]

        draw_box(image, box, label, confidence)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_detections(
    image: np.ndarray,
    image_name: str,
    boxes: List[List[int]],
    class_ids: List[int],
    scores: List[float],
    chosen_boxes: List[int],
) -> None:
    for i in chosen_boxes:
        box = boxes[i]

        label = str(classes[class_ids[i]])
        confidence = scores[i]

        draw_box(image, box, label, confidence)

    cv2.imwrite(f"results/images/{image_name}.png", image)
    cv2.destroyAllWindows()
