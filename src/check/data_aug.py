import cv2
import numpy as np
import random


def rotate_90cw(
    img: np.ndarray, 
    cls: np.ndarray, 
    x: np.ndarray, 
    y: np.ndarray, 
    w: np.ndarray, 
    h: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a square image 90° clockwise and adjust YOLO labels in normalized coords.

    Args:
        img: HxWxC numpy array.
        cls: array of class_id in [0,1].
        x: array of x_center in [0,1].
        y: array of y_center in [0,1].
        w: array of width in [0,1].
        h: array of height in [0,1].

    Returns:
        rot_img: Rotated image.
        new_labels: Nx5 array of adjusted labels.
    """
    rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    new_x = 1.0 - y
    new_y = x
    new_w = h
    new_h = w

    new_labels = np.hstack((cls, new_x, new_y, new_w, new_h))
    return rot_img, new_labels


def rotate_180(
    img: np.ndarray, 
    cls: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a square image 180° and adjust YOLO labels in normalized coords.

    Args:
        img: HxWxC numpy array.
        cls: array of class_id in [0,1].
        x: array of x_center in [0,1].
        y: array of y_center in [0,1].
        w: array of width in [0,1].
        h: array of height in [0,1].

    Returns:
        rot_img: Rotated image.
        new_labels: Nx5 array of adjusted labels.
    """
    rot_img = cv2.rotate(img, cv2.ROTATE_180)

    new_x = 1.0 - x
    new_y = 1.0 - y
    new_w = w
    new_h = h

    new_labels = np.hstack((cls, new_x, new_y, new_w, new_h))
    return rot_img, new_labels


def rotate_90ccw(
    img: np.ndarray,
    cls: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a square image 90° counter clockwise and adjust YOLO labels in normalized coords.

    Args:
        img: HxWxC numpy array.
        cls: array of class_id in [0,1].
        x: array of x_center in [0,1].
        y: array of y_center in [0,1].
        w: array of width in [0,1].
        h: array of height in [0,1].

    Returns:
        rot_img: Rotated image.
        new_labels: Nx5 array of adjusted labels.
    """
    rot_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    new_x = y
    new_y = 1.0 - x
    new_w = h
    new_h = w

    new_labels = np.hstack((cls, new_x, new_y, new_w, new_h))
    return rot_img, new_labels


def random_left_cut_and_pad(
    img: np.ndarray,
    labels: np.ndarray,
    max_frac: float = 1/3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly cut a vertical strip from the left (width ∈ [0, max_frac*W]),
    remove it, pad white on the right to restore original size, and adjust YOLO labels.

    Args:
      img: HxWxC numpy array (640x640x3).
      labels: Nx5 array [cls, x_c, y_c, w, h] normalized [0,1].
      max_frac: max fraction of width to cut (default 1/3).

    Returns:
      new_img: augmented image, same shape as img.
      new_labels: Nx5 array of adjusted labels.
    """

    cut_w = random.randint(0, int(max_frac * W))
    cropped = img[:, cut_w:]
    pad = np.ones((H, cut_w, 3), dtype=img.dtype) * 255
    new_img = np.hstack((cropped, pad))

    x_new = xp - cut_w

    # clamp center so box stays inside [0,W]
    x_new = np.clip(x_new, bw/2, W - bw/2)
    y_new = np.clip(y,    bh/2, H - bh/2)

    # normalize back
    x_n = x_new / W
    y_n = y_new / H
    w_n = bw    / W
    h_n = bh    / H

    new_labels = np.hstack((cls, x_n, y_n, w_n, h_n))
    return new_img, new_labels


if __name__ == "__main__":
    img = cv2.imread("page_042_asgm_part14_v0_png.rf.70e29144c1b529cba0ba8321cb17122e.jpg")
    labels = np.loadtxt("/home/hunglt31/examark/src/check/page_042_asgm_part14_v0_png.rf.70e29144c1b529cba0ba8321cb17122e.txt", ndmin=2)  
    
    cls = labels[:, 0:1]
    x = labels[:, 1:2]
    y = labels[:, 2:3]
    w = labels[:, 3:4]
    h = labels[:, 4:5]

    H, W = img.shape[:2]
    xp = labels[:, 1:2] * W
    yp = labels[:, 2:3] * H
    wp = labels[:, 3:4] * W
    hp = labels[:, 4:5] * H

    rot_90cw_img, rot_90cw_labels = rotate_90cw(img, cls, x, y, w, h)
    rot_180_img, rot_180_labels = rotate_180(img, cls, x, y, w, h)
    rot_90ccw_img, rot_90ccw_labels = rotate_90ccw(img, cls, x, y, w, h)

    cv2.imwrite("img_rot_90cw.jpg", rot_90cw_img)
    np.savetxt("img_rot_90cw.txt",
            rot_90cw_labels,
            fmt=["%d","%.8f","%.8f","%.8f","%.8f"])
    
    cv2.imwrite("img_rot_180.jpg", rot_180_img)
    np.savetxt("img_rot_180.txt",
            rot_180_labels,
            fmt=["%d","%.8f","%.8f","%.8f","%.8f"])
    
    cv2.imwrite("img_rot_90ccw.jpg", rot_90ccw_img)
    np.savetxt("img_rot_90ccw.txt",
            rot_90ccw_labels,
            fmt=["%d","%.8f","%.8f","%.8f","%.8f"])