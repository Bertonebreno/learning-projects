import cv2

from yolo.display import show_detections
from yolo.engine import apply_nms, forward, get_confident_boxes, print_net_architecture

if __name__ == "__main__":

    print_net_architecture()

    image_path = "images/city_scene.jpg"

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    output = forward(image)

    class_ids, scores, boxes = get_confident_boxes(output, height, width)

    chosen_boxes = apply_nms(boxes, scores)

    show_detections(image, boxes, class_ids, scores, chosen_boxes)
