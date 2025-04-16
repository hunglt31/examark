import os
import cv2
import numpy as np
import random
import albumentations as A

from glob import glob

# Constants
# Image size
WIDTH = 640
HEIGHT = 640


def rotate_90cw(
    img: np.ndarray, 
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a square image 90° clockwise and adjust YOLO labels in normalized coords.

    Args:
        img: HxWxC numpy array.
        labels: Nx5 array [cls, x_c, y_c, w, h] normalized [0,1].

    Returns:
        rot_img: Rotated image.
        new_labels: Nx5 array of adjusted labels.
    """

    rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    cls = labels[:, 0:1]
    x_norm = labels[:, 1:2]
    y_norm = labels[:, 2:3]
    w_norm = labels[:, 3:4]
    h_norm = labels[:, 4:5]

    new_x = 1.0 - y_norm
    new_y = x_norm
    new_w = h_norm
    new_h = w_norm

    new_labels = np.hstack((cls, new_x, new_y, new_w, new_h))
    return rot_img, new_labels


def rotate_180(
    img: np.ndarray, 
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a square image 180° and adjust YOLO labels in normalized coords.

    Args:
        img: HxWxC numpy array.
        labels: Nx5 array [cls, x_c, y_c, w, h] normalized [0,1].

    Returns:
        rot_img: Rotated image.
        new_labels: Nx5 array of adjusted labels.
    """

    rot_img = cv2.rotate(img, cv2.ROTATE_180)

    cls = labels[:, 0:1]
    x_norm = labels[:, 1:2]
    y_norm = labels[:, 2:3]
    w_norm = labels[:, 3:4]
    h_norm = labels[:, 4:5]

    new_x = 1.0 - x_norm
    new_y = 1.0 - y_norm
    new_w = w_norm
    new_h = h_norm

    new_labels = np.hstack((cls, new_x, new_y, new_w, new_h))
    return rot_img, new_labels


def rotate_90ccw(
    img: np.ndarray,
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a square image 90° counter clockwise and adjust YOLO labels in normalized coords.

    Args:
        img: HxWxC numpy array.
        labels: Nx5 array [cls, x_c, y_c, w, h] normalized [0,1].

    Returns:
        rot_img: Rotated image.
        new_labels: Nx5 array of adjusted labels.
    """

    rot_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cls = labels[:, 0:1]
    x_norm = labels[:, 1:2]
    y_norm = labels[:, 2:3]
    w_norm = labels[:, 3:4]
    h_norm = labels[:, 4:5]

    new_x = y_norm
    new_y = 1.0 - x_norm
    new_w = h_norm
    new_h = w_norm

    new_labels = np.hstack((cls, new_x, new_y, new_w, new_h))
    return rot_img, new_labels


def random_left_cut_and_pad(
    img: np.ndarray,
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly cut a vertical strip from the left,remove it, 
    pad white on the right to restore original size, and adjust YOLO labels.

    Args:
      img: HxWxC numpy array (640x640x3).
      labels: Nx5 array [cls, x_c, y_c, w, h] normalized [0,1].

    Returns:
      new_img: augmented image, same shape as img.
      new_labels: Nx5 array of adjusted labels.
    """

    cls = labels[:, 0:1]
    x_norm = labels[:, 1:2] 
    y_norm = labels[:, 2:3] 
    w_norm = labels[:, 3:4]
    h_norm = labels[:, 4:5] 

    cut_range = int(np.min(x_norm) * WIDTH - 100)
    cut_w = random.randint(cut_range // 2, cut_range)
    cropped = img[:, cut_w:]
    pad = np.ones((HEIGHT, cut_w, 3), dtype=img.dtype) * 255
    new_img = np.hstack((cropped, pad))

    x_norm = x_norm - cut_w / WIDTH

    new_labels = np.hstack((cls, x_norm, y_norm, w_norm, h_norm))
    return new_img, new_labels


def random_right_cut_and_pad(
    img: np.ndarray,
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly cut a vertical strip from the right, remove it, 
    pad white on the left to restore original size, and adjust YOLO labels.

    Args:
      img: HxWxC numpy array (640x640x3).
      labels: Nx5 array [cls, x_c, y_c, w, h] normalized [0,1].

    Returns:
      new_img: augmented image, same shape as img.
      new_labels: Nx5 array of adjusted labels.
    """

    cls = labels[:, 0:1]
    x_norm = labels[:, 1:2] 
    y_norm = labels[:, 2:3] 
    w_norm = labels[:, 3:4]
    h_norm = labels[:, 4:5] 

    cut_range = int(np.max(x_norm) * WIDTH + 100)
    cut_w = random.randint(cut_range, WIDTH - (WIDTH - cut_range) // 2)
    cropped = img[:, :cut_w]
    pad = np.ones((HEIGHT, WIDTH - cut_w, 3), dtype=img.dtype) * 255
    new_img = np.hstack((pad, cropped))

    x_norm = x_norm + (WIDTH - cut_w) / WIDTH

    new_labels = np.hstack((cls, x_norm, y_norm, w_norm, h_norm))
    return new_img, new_labels


def random_top_cut_and_pad(
    img: np.ndarray,
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly cut a horizontal strip from the top, remove it, 
    pad white on the bottom to restore original size, and adjust YOLO labels.

    Args:
      img: HxWxC numpy array (640x640x3).
      labels: Nx5 array [cls, x_c, y_c, w, h] normalized [0,1].

    Returns:
      new_img: augmented image, same shape as img.
      new_labels: Nx5 array of adjusted labels.
    """

    cls = labels[:, 0:1]
    x_norm = labels[:, 1:2] 
    y_norm = labels[:, 2:3] 
    w_norm = labels[:, 3:4]
    h_norm = labels[:, 4:5] 

    cut_range = int(np.min(y_norm) * HEIGHT - 100)
    cut_h = random.randint(cut_range // 2, cut_range)
    cropped = img[cut_h:, :]
    pad = np.ones((cut_h, WIDTH, 3), dtype=img.dtype) * 255
    new_img = np.vstack((cropped, pad))

    y_norm = y_norm - cut_h / HEIGHT

    new_labels = np.hstack((cls, x_norm, y_norm, w_norm, h_norm))
    return new_img, new_labels


def random_bottom_cut_and_pad(
    img: np.ndarray,
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly cut a horizontal strip from the bottom, remove it, 
    pad white on the top to restore original size, and adjust YOLO labels.

    Args:
      img: HxWxC numpy array (640x640x3).
      labels: Nx5 array [cls, x_c, y_c, w, h] normalized [0,1].

    Returns:
      new_img: augmented image, same shape as img.
      new_labels: Nx5 array of adjusted labels.
    """

    cls = labels[:, 0:1]
    x_norm = labels[:, 1:2] 
    y_norm = labels[:, 2:3] 
    w_norm = labels[:, 3:4]
    h_norm = labels[:, 4:5] 

    cut_range = int(np.max(y_norm) * HEIGHT + 100)
    cut_h = random.randint(cut_range, HEIGHT - (HEIGHT - cut_range) // 2)
    cropped = img[:cut_h, :]
    pad = np.ones((HEIGHT - cut_h, WIDTH, 3), dtype=img.dtype) * 255
    new_img = np.vstack((pad, cropped))

    y_norm = y_norm + (HEIGHT - cut_h) / HEIGHT

    new_labels = np.hstack((cls, x_norm, y_norm, w_norm, h_norm))
    return new_img, new_labels


transform = A.Compose([
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=50, val_shift_limit=25, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.08, p=1.0),
])

def augment_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)['image']
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
    return augmented


if __name__ == "__main__":
    input_root_dir =  "/home/hunglt31/examark/dataset/content"
    output_root_dir = "/home/hunglt31/examark/dataset1/content"
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(input_root_dir, split, 'images')
        label_dir = os.path.join(input_root_dir, split, 'labels')

        output_img_dir = os.path.join(output_root_dir, split, 'images')
        output_label_dir = os.path.join(output_root_dir, split, 'labels')

        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        for img_path in glob(os.path.join(img_dir, '*.jpg')):
            base_name = os.path.basename(img_path)[:-4]
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            img = cv2.imread(img_path)

            augment_img = augment_color(img)
            cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_aug.png"), augment_img)
            labels = np.loadtxt(label_path, delimiter=" ")
            np.savetxt(os.path.join(output_label_dir, f"{base_name}_aug.txt"),
                       labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
            try: 
                if "student_id" in base_name or "exam_id" in base_name:
                    pad_left_img, pad_left_labels = random_left_cut_and_pad(img, labels)
                    pad_left_img_aug = augment_color(pad_left_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_pad_left_aug.png"), pad_left_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_pad_left_aug.txt"),
                                pad_left_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    pad_right_img, pad_right_labels = random_right_cut_and_pad(img, labels)
                    pad_right_img_aug = augment_color(pad_right_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_pad_right_aug.png"), pad_right_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_pad_right_aug.txt"),
                                pad_right_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])

                    rot_90cw_img, rot_90cw_labels = rotate_90cw(img, labels)
                    rot_90cw_img_aug = augment_color(rot_90cw_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90cw_aug.png"), rot_90cw_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90cw_aug.txt"),
                                rot_90cw_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])

                    pad_top_img, pad_top_labels = random_top_cut_and_pad(rot_90cw_img, rot_90cw_labels)
                    pad_top_img_aug = augment_color(pad_top_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_pad_top_aug.png"), pad_top_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_pad_top_aug.txt"),
                                pad_top_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    pad_bottom_img, pad_bottom_labels = random_bottom_cut_and_pad(rot_90cw_img, rot_90cw_labels)
                    pad_bottom_img_aug = augment_color(pad_bottom_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_pad_bottom_aug.png"), pad_bottom_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_pad_bottom_aug.txt"),   
                                pad_bottom_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                elif "part1" in base_name:
                    pad_top_img, pad_top_labels = random_top_cut_and_pad(img, labels)
                    pad_top_img_aug = augment_color(pad_top_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_pad_top_aug.png"), pad_top_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_pad_top_aug.txt"),
                                pad_top_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    pad_bottom_img, pad_bottom_labels = random_bottom_cut_and_pad(img, labels)
                    pad_bottom_img_aug = augment_color(pad_bottom_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_pad_bottom_aug.png"), pad_bottom_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_pad_bottom_aug.txt"),
                                pad_bottom_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_90cw_img, rot_90cw_labels = rotate_90cw(img, labels)
                    rot_90cw_img_aug = augment_color(rot_90cw_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90cw_aug.png"), rot_90cw_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90cw_aug.txt"),
                                rot_90cw_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])        
                    
                    rot_90cw_pad_left_img, rot_90cw_pad_left_labels = random_left_cut_and_pad(rot_90cw_img, rot_90cw_labels)
                    rot_90cw_pad_left_img_aug = augment_color(rot_90cw_pad_left_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90cw_pad_left_aug.png"), rot_90cw_pad_left_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90cw_pad_left_aug.txt"),
                                rot_90cw_pad_left_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])

                    rot_90cw_pad_right_img, rot_90cw_pad_right_labels = random_right_cut_and_pad(rot_90cw_img, rot_90cw_labels)
                    rot_90cw_pad_right_img_aug = augment_color(rot_90cw_pad_right_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90cw_pad_right_aug.png"), rot_90cw_pad_right_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90cw_pad_right_aug.txt"),
                                rot_90cw_pad_right_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_180_img, rot_180_labels = rotate_180(img, labels)
                    rot_180_img_aug = augment_color(rot_180_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_180_aug.png"), rot_180_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_180_aug.txt"),
                                rot_180_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_180_pad_top_img, rot_180_pad_top_labels = random_top_cut_and_pad(rot_180_img, rot_180_labels)
                    rot_180_pad_top_img_aug = augment_color(rot_180_pad_top_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_180_pad_top_aug.png"), rot_180_pad_top_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_180_pad_top_aug.txt"),
                                rot_180_pad_top_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_180_pad_bottom_img, rot_180_pad_bottom_labels = random_bottom_cut_and_pad(rot_180_img, rot_180_labels)
                    rot_180_pad_bottom_img_aug = augment_color(rot_180_pad_bottom_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_180_pad_bottom_aug.png"), rot_180_pad_bottom_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_180_pad_bottom_aug.txt"),
                                rot_180_pad_bottom_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_90ccw_img, rot_90ccw_labels = rotate_90ccw(img, labels)
                    rot_90ccw_img_aug = augment_color(rot_90ccw_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90ccw_aug.png"), rot_90ccw_img_aug)      
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90ccw_aug.txt"),
                                rot_90ccw_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_90ccw_pad_left_img, rot_90ccw_pad_left_labels = random_left_cut_and_pad(rot_90ccw_img, rot_90ccw_labels)
                    rot_90ccw_pad_left_img_aug = augment_color(rot_90ccw_pad_left_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90ccw_pad_left_aug.png"), rot_90ccw_pad_left_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90ccw_pad_left_aug.txt"),
                                rot_90ccw_pad_left_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_90ccw_pad_right_img, rot_90ccw_pad_right_labels = random_right_cut_and_pad(rot_90ccw_img, rot_90ccw_labels)
                    rot_90ccw_pad_right_img_aug = augment_color(rot_90ccw_pad_right_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90ccw_pad_right_aug.png"), rot_90ccw_pad_right_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90ccw_pad_right_aug.txt"),
                                rot_90ccw_pad_right_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                else:
                    rot_90cw_img, rot_90cw_labels = rotate_90cw(img, labels)
                    rot_90cw_img_aug = augment_color(rot_90cw_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90cw_aug.png"), rot_90cw_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90cw_aug.txt"),
                                rot_90cw_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_180_img, rot_180_labels = rotate_180(img, labels)
                    rot_180_img_aug = augment_color(rot_180_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_180_aug.png"), rot_180_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_180_aug.txt"),
                                rot_180_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
                    
                    rot_90ccw_img, rot_90ccw_labels = rotate_90ccw(img, labels)
                    rot_90ccw_img_aug = augment_color(rot_90ccw_img)
                    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_rot_90ccw_aug.png"), rot_90ccw_img_aug)
                    np.savetxt(os.path.join(output_label_dir, f"{base_name}_rot_90ccw_aug.txt"),
                                rot_90ccw_labels, fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])
            
            except Exception as e:
                print(f"Skipping {base_name} due to error: {e}")
                continue
            print(f"Processed {base_name} in {split} set.")
        print(f"Finished processing {split} set.")
    print("All done!")
