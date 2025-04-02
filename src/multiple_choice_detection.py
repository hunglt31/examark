import cv2
import numpy as np
from ultralytics import YOLO
from support_functions import *
from sklearn.cluster import KMeans
import pandas as pd
import csv 


model = YOLO('../model/yolo11m_best_model.pt')


def padding_image(image):
    image_w = image.shape[1]
    image_h = image.shape[0]

    if image_w > 640 or image_h > 640:
        if image_w > image_h:
            new_image_w = 640
            new_image_h = int(image_h / image_w * 640)
        else:
            new_image_h = 640
            new_image_w = int(image_w / image_h * 640)
        image = cv2.resize(image, (new_image_w, new_image_h))
    else:
        new_image_w = image_w
        new_image_h = image_h

    pad_w = 640 - new_image_w
    pad_h = 640 - new_image_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return image


def inference(image, part):
    image = padding_image(image)
    results = model(image)

    list_boxes = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            class_id = class_ids[i]
            box = boxes[i]
            x_min, y_min, x_max, y_max = box.xyxy[0]
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            class_id = int(class_ids[i])
            if part == 0:
                if 23 < x_max - x_min < 43 and 26 < y_max - y_min < 46: 
                    list_boxes.append((center_x, center_y, int(class_id))) 
            elif part == 1 or part == 2:
                if 32 < x_max - x_min < 52 and 32 < y_max - y_min < 52: 
                    list_boxes.append((center_x, center_y, int(class_id))) 
            else:
                if 26 < x_max - x_min < 46 and 26 < y_max - y_min < 46: 
                    list_boxes.append((center_x, center_y, int(class_id))) 

    return list_boxes


def kmeans_processing(list_boxes, num_center_x, num_center_y):
    list_x = np.array(list_boxes)[:, 0]
    list_y = np.array(list_boxes)[:, 1]

    kmeans_x = KMeans(n_clusters=num_center_x, random_state=0).fit(list_x.reshape(-1, 1))
    kmeans_y = KMeans(n_clusters=num_center_y, random_state=0).fit(list_y.reshape(-1, 1))

    array_boxes = np.zeros((num_center_y, num_center_x))

    sorted_centers_x = np.sort(kmeans_x.cluster_centers_.flatten())
    sorted_centers_y = np.sort(kmeans_y.cluster_centers_.flatten())

    for box in list_boxes:
        new_point = box[:2]
        new_point_class_id = box[2]

        center_idx_x = np.argmin(np.abs(sorted_centers_x - new_point[0]))
        center_idx_y = np.argmin(np.abs(sorted_centers_y - new_point[1]))

        if new_point_class_id == 0:
            array_boxes[center_idx_y, center_idx_x] = 1
        else:
            array_boxes[center_idx_y, center_idx_x] = 0
    
    return array_boxes


def grading_assignment(image_path):
  image = cv2.imread(image_path)
  image_name = os.path.splitext(os.path.basename(image_path))[0]

  SBD_part1, SBD_part2, assignment_part11, assignment_part12, assignment_part13, assignment_part14, assignment_part21, assignment_part22, assignment_part23, assignment_part24, assignment_part31, assignment_part32, assignment_part33, assignment_part34, assignment_part35, assignment_part36 = split_image(image)

  # Get info
  lbi_SBD = inference(SBD_part1, 0)
  lbi_MaDe = inference(SBD_part2, 0)

  arr_SBD = kmeans_processing(lbi_SBD, 6, 10)
  arr_MaDe = kmeans_processing(lbi_MaDe, 3, 10)
  
  SBD, MaDe = process_info_part(np.hstack((arr_SBD, arr_MaDe)))

  # Get result part 1
  lba_part11 = inference(assignment_part11, 1)
  lba_part12 = inference(assignment_part12, 1)
  lba_part13 = inference(assignment_part13, 1)
  lba_part14 = inference(assignment_part14, 1)

  arr_part11 = kmeans_processing(lba_part11, 4, 10)
  arr_part12 = kmeans_processing(lba_part12, 4, 10)
  arr_part13 = kmeans_processing(lba_part13, 4, 10)
  arr_part14 = kmeans_processing(lba_part14, 4, 10)

  array_part1 = np.vstack((arr_part11, arr_part12, arr_part13, arr_part14))
  result_part1 = process_assignment_part1(array_part1)

  # Get result part 2
  lba_part21 = inference(assignment_part21, 2)
  lba_part22 = inference(assignment_part22, 2)
  lba_part23 = inference(assignment_part23, 2)
  lba_part24 = inference(assignment_part24, 2)+

  arr_part21 = kmeans_processing(lba_part21, 4, 4)
  arr_part22 = kmeans_processing(lba_part22, 4, 4)
  arr_part23 = kmeans_processing(lba_part23, 4, 4)
  arr_part24 = kmeans_processing(lba_part24, 4, 4)

  array_part2 = np.hstack((arr_part21, arr_part22, arr_part23, arr_part24))
  result_part2 = process_assignment_part2(array_part2) 
                                          
  # Get result part 3
  lba_part31 = inference(assignment_part31, 3)
  lba_part32 = inference(assignment_part32, 3)
  lba_part33 = inference(assignment_part33, 3)
  lba_part34 = inference(assignment_part34, 3)
  lba_part35 = inference(assignment_part35, 3)
  lba_part36 = inference(assignment_part36, 3)

  arr_part31 = kmeans_processing(lba_part31, 4, 12)
  arr_part32 = kmeans_processing(lba_part32, 4, 12)
  arr_part33 = kmeans_processing(lba_part33, 4, 12)
  arr_part34 = kmeans_processing(lba_part34, 4, 12)
  arr_part35 = kmeans_processing(lba_part35, 4, 12)
  arr_part36 = kmeans_processing(lba_part36, 4, 12)

  matrices_part3 = [arr_part31, arr_part32, arr_part33, arr_part34, arr_part35, arr_part36]
  result_part3 = [process_part3(matrix) for matrix in matrices_part3]

  return [str(image_name), SBD, MaDe, "Answer"] + result_part1 + result_part2 + result_part3


def create_headers():
    header1 = ['Exam', 'Info', 'Assignment', 'Part'] + [1] * 40 + [2] * 8 + [3] * 6

    header2 = ['Image', 'SBD', 'MaDe', 'Question']
    
    for i in range(1, 4):
        num_questions = 40 if i == 1 else 8 if i == 2 else 6 
        for q in range(1, num_questions + 1):
            header2.append(f'{q}')

    return [header1, header2]


def write_result_to_csv(results, file_name, create_header):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        headers = create_headers()
        columns = headers + results
        rows = zip(*columns)
        writer.writerows(rows) 


def process_image(image_path, path_result):
    try:
        exam_result = grading_assignment(image_path)
        return exam_result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None