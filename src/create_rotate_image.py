import cv2
import numpy as np
import random

def rotate_and_draw(image_path, angle=2, line_color=(0, 0, 0), circle_color=(0, 0, 0), circle_radius=20, num_circles=100):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    # Compute rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Compute new bounding dimensions after rotation
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix to shift the image center
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Rotate image and pad to maintain rectangular shape
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderValue=(255, 255, 255))
    
    # Draw 100 random circles with size 28x28 pixels
    for _ in range(num_circles):
        center_x = random.randint(circle_radius, new_w - circle_radius)
        center_y = random.randint(circle_radius, new_h - circle_radius)
        cv2.circle(rotated_image, (center_x, center_y), circle_radius, circle_color, -1)
    
    return rotated_image

# Example usage
image_path = "refImg.png"
output_path = "image1.png"
result = rotate_and_draw(image_path, circle_radius=14, num_circles=100)  # Circle radius is half of 28
cv2.imwrite(output_path, result)
print("Image saved as", output_path)
