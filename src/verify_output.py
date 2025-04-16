from ultralytics import YOLO
import cv2, os
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import copy
from PIL import Image

def load_ground_truth_label(data):
    class_ids = data[:, 0].astype(int)
    x_centers = data[:, 1].reshape(-1, 1)
    y_centers = data[:, 2].reshape(-1, 1)

    db_x = DBSCAN(eps=0.05, min_samples=1).fit(x_centers)
    x_labels = db_x.labels_

    db_y = DBSCAN(eps=0.05, min_samples=1).fit(y_centers)
    y_labels = db_y.labels_

    unique_x = np.unique(x_labels)
    x_cluster_means = []
    for label in unique_x:
        idx = np.where(x_labels == label)[0]
        mean_val = np.mean(x_centers[idx])
        x_cluster_means.append((label, mean_val))
    x_cluster_means.sort(key=lambda x: x[1])
    ordered_x_labels = [t[0] for t in x_cluster_means]

    unique_y = np.unique(y_labels)
    y_cluster_means = []
    for label in unique_y:
        idx = np.where(y_labels == label)[0]
        mean_val = np.mean(y_centers[idx])
        y_cluster_means.append((label, mean_val))
    y_cluster_means.sort(key=lambda x: x[1])
    ordered_y_labels = [t[0] for t in y_cluster_means]

    n_cols = len(ordered_x_labels)   
    n_rows = len(ordered_y_labels)   

    matrix = np.full((n_rows, n_cols), 1, dtype=int)

    cell_dict = {}
    for i in range(len(data)):
        x_lab = x_labels[i]
        y_lab = y_labels[i]
        try:
            col_index = ordered_x_labels.index(x_lab)
            row_index = ordered_y_labels.index(y_lab)
        except ValueError:
            continue
        cell_dict.setdefault((row_index, col_index), []).append(class_ids[i])

    for (r, c), ids in cell_dict.items():
        majority = Counter(ids).most_common(1)[0][0]
        matrix[r, c] = majority

    return matrix


def load_ground_truth_pred(data):
    class_ids = data[:, 0].astype(int)
    x_centers = data[:, 1].reshape(-1, 1)
    y_centers = data[:, 2].reshape(-1, 1)

    db_x = DBSCAN(eps=5, min_samples=1).fit(x_centers)
    x_labels = db_x.labels_

    db_y = DBSCAN(eps=5, min_samples=1).fit(y_centers)
    y_labels = db_y.labels_

    unique_x = np.unique(x_labels)
    x_cluster_means = []
    for label in unique_x:
        idx = np.where(x_labels == label)[0]
        mean_val = np.mean(x_centers[idx])
        x_cluster_means.append((label, mean_val))
    x_cluster_means.sort(key=lambda x: x[1])
    ordered_x_labels = [t[0] for t in x_cluster_means]

    unique_y = np.unique(y_labels)
    y_cluster_means = []
    for label in unique_y:
        idx = np.where(y_labels == label)[0]
        mean_val = np.mean(y_centers[idx])
        y_cluster_means.append((label, mean_val))
    y_cluster_means.sort(key=lambda x: x[1])
    ordered_y_labels = [t[0] for t in y_cluster_means]

    n_cols = len(ordered_x_labels)   
    n_rows = len(ordered_y_labels)   

    matrix = np.full((n_rows, n_cols), 1, dtype=int)

    cell_dict = {}
    for i in range(len(data)):
        x_lab = x_labels[i]
        y_lab = y_labels[i]
        try:
            col_index = ordered_x_labels.index(x_lab)
            row_index = ordered_y_labels.index(y_lab)
        except ValueError:
            continue
        cell_dict.setdefault((row_index, col_index), []).append(class_ids[i])

    for (r, c), ids in cell_dict.items():
        majority = Counter(ids).most_common(1)[0][0]
        matrix[r, c] = majority

    return matrix


def inference(model, img):
    """
    Runs inference on the image, draws boxes, computes center positions,
    flips classes (0 becomes 1, 1 becomes 0) and clusters boxes using DBSCAN.
    Returns the processed image and an overall predicted label via majority vote.
    """
    results = model(img)
    list_boxes = []

    for result in results:
        # Iterate over each detected box
        for box in result.boxes:
            # Extract bounding box coordinates, confidence and predicted class
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            class_id = int(box.cls.cpu().numpy()[0])
            if class_id == 0 or class_id == 1:
                list_boxes.append([class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    return list_boxes


def draw_boxes(img, boxes, is_gt=False):
    img_copy = img.copy()
    for box in boxes:
        class_id, cx, cy, w, h = box
        if is_gt:
            cx *= 640
            cy *= 640
            w *= 640
            h *= 640
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        color = (0, 0, 255) if class_id == 0 else (255, 0, 0)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
    return img_copy


if __name__ == "__main__":
    # # Load YOLOv8 model
    # model_path = "../models/metadata_yolov8m/weights/best.pt"
    # metadata_model = YOLO(model_path)
    
    # # Define input and output directories
    # input_img_dir = "../dataset1/metadata/valid/images"
    # input_lbl_dir = "../dataset1/metadata/valid/labels"
    # output_img_dir = "../validate_result/metadata_miss_matches"
    # os.makedirs(output_img_dir, exist_ok=True)
    
    # for img_name in os.listdir(input_img_dir):
    #     base_name = base_name = os.path.basename(img_name)[:-4]
    #     print(base_name)
    #     img = cv2.imread(os.path.join(input_img_dir, img_name))
    #     gt_labels = np.loadtxt(os.path.join(input_lbl_dir, base_name + ".txt"), delimiter=' ', ndmin=2)
    #     gt_labels_matrix = load_ground_truth_label(gt_labels).astype(int)

    #     pred_boxes = inference(metadata_model, img)
    #     pred_boxes_matrix = load_ground_truth_pred(np.array(pred_boxes)).astype(int)
    #     print("GT matrix:\n", gt_labels_matrix)
    #     print("Pred matrix:\n", pred_boxes_matrix)

    #     if np.any(pred_boxes_matrix != gt_labels_matrix):
    #         img_gt = draw_boxes(img, gt_labels, True)
    #         img_pred = draw_boxes(img, pred_boxes)
            
    #         combined = np.hstack((img_gt, img_pred))
    #         cv2.imwrite(os.path.join(output_img_dir, base_name + ".jpg"), combined)


    # Load YOLOv8 model
    model_path = "../models/content_yolov8m/weights/best.pt"
    content_model = YOLO(model_path)
    
    # Define input and output directories
    input_img_dir = "../dataset1/content/valid/images"
    input_lbl_dir = "../dataset1/content/valid/labels"
    output_img_dir = "../validate_result/content_miss_matches"
    os.makedirs(output_img_dir, exist_ok=True)
    
    for img_name in os.listdir(input_img_dir):
        base_name = base_name = os.path.basename(img_name)[:-4]
        print(base_name)
        img = cv2.imread(os.path.join(input_img_dir, img_name))
        gt_labels = np.loadtxt(os.path.join(input_lbl_dir, base_name + ".txt"), delimiter=' ', ndmin=2)
        gt_labels_matrix = load_ground_truth_label(gt_labels).astype(int)

        pred_boxes = inference(content_model, img)
        pred_boxes_matrix = load_ground_truth_pred(np.array(pred_boxes)).astype(int)
        # print("GT matrix:\n", gt_labels_matrix)
        # print("Pred matrix:\n", pred_boxes_matrix)

        if not np.array_equal(pred_boxes_matrix, gt_labels_matrix):
            print("Mismatch detected!")
            img_gt = draw_boxes(img, gt_labels, True)
            img_pred = draw_boxes(img, pred_boxes)
            
            combined = np.hstack((img_gt, img_pred))
            cv2.imwrite(os.path.join(output_img_dir, base_name + ".jpg"), combined)
