from typing import List, Sequence, Tuple

import cv2
import numpy as np


def print_net_architecture():
    net = cv2.dnn.readNetFromDarknet(
        cfgFile="files/yolov3.cfg",
        darknetModel="files/yolov3.weights",
    )

    layer_ids = [net.getLayerId(layer_name) for layer_name in net.getLayerNames()]

    for id in layer_ids:
        layer = net.getLayer(id)
        if layer.type in ["ReLU", "BatchNorm", "Permute", "Identity", "Concat"]:
            continue

        if layer.type == "Region":
            print("YOLO detection")
            continue

        if layer.type == "Eltwise":
            print("Shortcut Layer")
            continue

        if layer.type == "Resize":
            print("Upsample")
            continue

        try:
            print(layer.type, np.shape(layer.blobs))
        except ValueError:
            print("Output de detecção")


def forward(image: np.ndarray) -> Sequence[cv2.typing.MatLike]:
    net = cv2.dnn.readNetFromDarknet(
        cfgFile="files/yolov3.cfg",
        darknetModel="files/yolov3.weights",
    )

    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1 / 255.0, size=(608, 608), swapRB=True, crop=False
    )

    names = net.getLayerNames()
    output_layers = [names[i - 1] for i in net.getUnconnectedOutLayers()]

    net.setInput(blob)
    outs = net.forward(output_layers)

    return outs


def get_confident_boxes(
    net_output: Sequence[np.ndarray],
    image_height: int,
    image_width: int,
    min_confidence=0.5,
) -> Tuple[List[int], List[float], List[List[int]]]:
    class_ids = []
    scores = []
    boxes = []

    for out in net_output:
        for detection in out:
            all_scores = detection[5:]
            class_id = np.argmax(all_scores)
            score = all_scores[class_id]

            if score > min_confidence:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                scores.append(float(score))
                class_ids.append(class_id)

    return [int(i) for i in class_ids], scores, boxes


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)

    intersection_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def apply_nms(boxes, confidences, iou_threshold=0.4):
    indices = []
    boxes = np.array(boxes)
    confidences = np.array(confidences)

    sorted_indices = np.argsort(confidences)[::-1]

    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        indices.append(current_index)

        remaining_indices = sorted_indices[1:]

        filtered_indices = []
        for i in remaining_indices:
            iou = calculate_iou(boxes[current_index], boxes[i])
            if iou < iou_threshold:
                filtered_indices.append(i)

        sorted_indices = filtered_indices

    return indices
