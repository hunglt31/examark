import cv2
import numpy as np

def draw_bounding_boxes(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on the image based on labels.

    Args:
      image: HxWxC numpy array.
      labels: Nx5 array of [class_id, x_center, y_center, width, height] with values normalized between [0, 1].

    Returns:
      img_with_boxes: Image with bounding boxes drawn.
    """
    img_with_boxes = image.copy()
    h, w = img_with_boxes.shape[:2]

    for label in labels:
        class_id, x_center, y_center, width, height = label
        class_id = int(class_id)
        x_center, y_center, width, height = (x_center * w, y_center * h, width * w, height * h)

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        if class_id == 0:
            color = (0, 0, 255)  
        elif class_id == 1:
            color = (255, 0, 0) 
        else:
            color = (0, 255, 0) 

        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)

    return img_with_boxes

if __name__ == "__main__":
    image_path = "img_rot_90cw.jpg"
    image = cv2.imread(image_path)
    labels = np.loadtxt("img_rot_90cw.txt", delimiter=" ")
    img_with_boxes = draw_bounding_boxes(image, labels)
    cv2.imwrite("rotated_90cw_output_boxes.png", img_with_boxes)

    image_path = "img_rot_180.jpg"
    image = cv2.imread(image_path)
    labels = np.loadtxt("img_rot_180.txt", delimiter=" ")
    img_with_boxes = draw_bounding_boxes(image, labels)
    cv2.imwrite("rotated_180_output_boxes.png", img_with_boxes)

    image_path = "img_rot_90ccw.jpg"
    image = cv2.imread(image_path)
    labels = np.loadtxt("img_rot_90ccw.txt", delimiter=" ")
    img_with_boxes = draw_bounding_boxes(image, labels)
    cv2.imwrite("rotated_90ccw_output_boxes.png", img_with_boxes)

    image_path = "img_pad_top.jpg"
    image = cv2.imread(image_path)
    labels = np.loadtxt("img_pad_top.txt", delimiter=" ")
    img_with_boxes = draw_bounding_boxes(image, labels)
    cv2.imwrite("img_pad_top_output_boxes.png", img_with_boxes)

    image_path = "img_pad_bottom.jpg"
    image = cv2.imread(image_path)
    labels = np.loadtxt("img_pad_bottom.txt", delimiter=" ")
    img_with_boxes = draw_bounding_boxes(image, labels)
    cv2.imwrite("img_pad_bottom_output_boxes.png", img_with_boxes)

    image_path = "img_pad_left.jpg"
    image = cv2.imread(image_path)
    labels = np.loadtxt("img_pad_left.txt", delimiter=" ")
    img_with_boxes = draw_bounding_boxes(image, labels)
    cv2.imwrite("img_pad_left_output_boxes.png", img_with_boxes)

    image_path = "img_pad_right.jpg"
    image = cv2.imread(image_path)
    labels = np.loadtxt("img_pad_right.txt", delimiter=" ")
    img_with_boxes = draw_bounding_boxes(image, labels)
    cv2.imwrite("img_pad_right_output_boxes.png", img_with_boxes)