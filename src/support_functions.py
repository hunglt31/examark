import cv2, os
import numpy as np
from imutils.object_detection import non_max_suppression
from sklearn.cluster import KMeans, DBSCAN
import collections
import pandas as pd
import ast


def straighten_image(image_path, image_name, output_dir, initial_threshold=0.86, max_iterations=500):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (2255, 3151))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('../template/large_corner_v6.png', cv2.IMREAD_GRAYSCALE)
    print(f"Straightening image {image_name}.jpg ...")
    
    threshold, iteration = initial_threshold, 0
    unique_corners = []

    while iteration < max_iterations:
        result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        corners = sorted(zip(*loc[::-1]), key=lambda x: (x[1], x[0]))

        unique_corners = []
        for pt in corners:
            if not unique_corners or np.linalg.norm(np.array(pt) - np.array(unique_corners[-1])) > 50:
                unique_corners.append(pt)

        if len(unique_corners) == 4:
            break
        threshold += -0.001 if len(unique_corners) < 4 else 0.001
        iteration += 1

    if len(unique_corners) != 4:
        cv2.imwrite(f"{output_dir}/{image_name}_original.png", image)
        print(f"Can't straightening image, threshold = {threshold}.")
        return image

    unique_corners = sorted(unique_corners, key=lambda pt: (pt[1], pt[0]))
    top_left, top_right = sorted(unique_corners[:2], key=lambda pt: pt[0])
    bottom_left, bottom_right = sorted(unique_corners[2:], key=lambda pt: pt[0])
    
    top_left = (top_left[0] - 100, top_left[1] - 150)
    top_right = (top_right[0] + 200, top_right[1] - 150)
    bottom_left = (bottom_left[0] - 100, bottom_left[1] + 300)
    bottom_right = (bottom_right[0] + 200, bottom_right[1] + 300)
    
    mask_image = image.copy()
    for (x, y) in unique_corners:
        cv2.circle(mask_image, (x, y), 20, (0, 0, 255), -1)
    cv2.imwrite(f"./mask/{image_name}.png", mask_image)
    
    src_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")
    dst_points = np.array([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    straightened_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    cv2.imwrite(f"{output_dir}/{image_name}_straightened.png", straightened_image)
    print(f"Straightening successfully with threshold {threshold}.")
    

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def split_image(image):
    image = cv2.resize(image, (2255, 3151))

    tmplt1 = cv2.imread("../template/large_corner_v6.png")
    tmplt2 = cv2.imread("../template/check_multiple_template_matching_6.png")

    hh_1, ww_1 = tmplt1.shape[:2]
    hh_2, ww_2 = tmplt2.shape[:2]

    result_1 = cv2.matchTemplate(image, tmplt1, cv2.TM_SQDIFF)
    result_2 = cv2.matchTemplate(image, tmplt2, cv2.TM_SQDIFF)

    num_large = 0
    num_small = 0
    add_thresh_large = 0
    add_thresh_small = 0

    blacklist = []
    while num_large != 4:
        thresh1 = 100000000 + add_thresh_large
        (yCoords_1, xCoords_1) = np.where(result_1 <= thresh1)
        rects_1 = []

        for (x, y) in zip(xCoords_1, yCoords_1):
            rects_1.append((x, y, x + ww_1, y + hh_1))

        if len(rects_1) > 0:
            pick_1 = non_max_suppression(np.array(rects_1), overlapThresh=0.5)
        else:
            pick_1 = []

        large_contours_matrix = np.array(pick_1)
        num_large = len(large_contours_matrix)

        add_thresh_large += 5000000

    while num_small != 27:
        thresh2 = 10000000 + add_thresh_small
        (yCoords_2, xCoords_2) = np.where(result_2 <= thresh2)
        rects_2 = []

        for (x, y) in zip(xCoords_2, yCoords_2):
            rects_2.append((x, y, x + ww_2, y + hh_2))

        if len(rects_2) == 0:
            add_thresh_small += 1000
            continue
        else:
            pick_2 = non_max_suppression(np.array(rects_2), overlapThresh=0.5)

        if len(pick_2) == 0:
            add_thresh_small += 1000
            continue

        if len(blacklist) > 0:
            indices_to_remove = []
            for idx, rect in enumerate(pick_2):
                for bl_rect in blacklist:
                    overlap = compute_iou(rect, bl_rect)
                    if overlap > 0.8:
                        indices_to_remove.append(idx)
                        break  
            pick_2 = np.delete(pick_2, indices_to_remove, axis=0)

        num_small = len(pick_2)
        if num_small == 27:
            x1_list = pick_2[:, 1] 
            dbscan = DBSCAN(eps=20, min_samples=1)
            labels = dbscan.fit_predict(x1_list.reshape(-1, 1))

            num_clusters = len(set(labels))

            if num_clusters != 6:
                unique, counts = np.unique(labels, return_counts=True)
                cluster_counts = dict(zip(unique, counts))

                for rect, label in zip(pick_2, labels):
                    x1 = rect[1] 
                    if x1 > 1080 and cluster_counts[label] == 1:
                        if not any(np.array_equal(rect, bl_rect) for bl_rect in blacklist):
                            blacklist.append(rect)
                            num_small -= 1

                add_thresh_small += 10000
                continue
            else:
                small_contours_matrix = np.array(pick_2)
                break
        else:
            add_thresh_small += 10000
            continue

    large_contours_matrix = np.array(pick_1)
    small_contours_matrix = np.array(pick_2)

    large_contours_matrix = large_contours_matrix[large_contours_matrix[:,1].argsort()]
    contours_top_matrix = large_contours_matrix[:2]
    contours_top_matrix = contours_top_matrix[contours_top_matrix[:, 0].argsort()]
    top_left_contours = contours_top_matrix[0]
    top_right_contours = contours_top_matrix[1]

    contours_bot_matrix = large_contours_matrix[-2:]
    contours_bot_matrix = contours_bot_matrix[contours_bot_matrix[:, 0].argsort()]
    bot_left_contours = contours_bot_matrix[0]
    bot_right_contours = contours_bot_matrix[1]

    small_contours_matrix = small_contours_matrix[small_contours_matrix[:,1].argsort()]
    contours_top_1_matrix = small_contours_matrix[:2]
    contours_top_1_matrix = contours_top_1_matrix[contours_top_1_matrix[:, 0].argsort()]
    top_1_left_contours = contours_top_1_matrix[0]
    top_1_right_contours = contours_top_1_matrix[1]

    contours_top_2_matrix = small_contours_matrix[2:4]
    contours_top_2_matrix = contours_top_2_matrix[contours_top_2_matrix[:, 0].argsort()]
    top_2_left_contours = contours_top_2_matrix[0]
    top_2_right_contours = contours_top_2_matrix[1]

    contours_top_3_matrix = small_contours_matrix[4:8]
    contours_top_3_matrix = contours_top_3_matrix[contours_top_3_matrix[:, 0].argsort()]
    top_3_1_contours = contours_top_3_matrix[0]
    top_3_2_contours = contours_top_3_matrix[1]
    top_3_3_contours = contours_top_3_matrix[2]
    top_3_4_contours = contours_top_3_matrix[3]

    contours_top_4_matrix = small_contours_matrix[8:13]
    contours_top_4_matrix = contours_top_4_matrix[contours_top_4_matrix[:, 0].argsort()]
    top_4_1_contours = contours_top_4_matrix[0]
    top_4_2_contours = contours_top_4_matrix[1]
    top_4_3_contours = contours_top_4_matrix[2]
    top_4_4_contours = contours_top_4_matrix[3]
    top_4_5_contours = contours_top_4_matrix[4]

    contours_top_5_matrix = small_contours_matrix[13:22]
    contours_top_5_matrix = contours_top_5_matrix[contours_top_5_matrix[:, 0].argsort()]
    top_5_1_contours = contours_top_5_matrix[0]
    top_5_2_contours = contours_top_5_matrix[1]
    top_5_3_contours = contours_top_5_matrix[2]
    top_5_4_contours = contours_top_5_matrix[3]
    top_5_5_contours = contours_top_5_matrix[4]
    top_5_6_contours = contours_top_5_matrix[5]
    top_5_7_contours = contours_top_5_matrix[6]
    top_5_8_contours = contours_top_5_matrix[7]
    top_5_9_contours = contours_top_5_matrix[8]

    contours_top_6_matrix = small_contours_matrix[22:27]
    contours_top_6_matrix = contours_top_6_matrix[contours_top_6_matrix[:, 0].argsort()]
    top_6_1_contours = contours_top_6_matrix[0]
    top_6_2_contours = contours_top_6_matrix[1]
    top_6_3_contours = contours_top_6_matrix[2]
    top_6_4_contours = contours_top_6_matrix[3]
    top_6_5_contours = contours_top_6_matrix[4]

    coords_dict = {}
    SBD_part1_top = top_1_left_contours[3]
    SBD_part1_left = (top_1_left_contours[2] + top_1_left_contours[0]) // 2
    coords_dict['SBD_part1'] = [int(SBD_part1_left), int(SBD_part1_top)]

    SBD_part2_top = top_1_right_contours[3]
    SBD_part2_left = (top_1_right_contours[2] + top_1_right_contours[0]) // 2
    coords_dict['SBD_part2'] = [int(SBD_part2_left), int(SBD_part2_top)]

    assignment_part11_top = top_3_1_contours[3]
    assignment_part11_left = (top_3_1_contours[2] + top_3_1_contours[0]) // 2
    coords_dict['assignment_part11'] = [int(assignment_part11_left), int(assignment_part11_top)]

    assignment_part12_top = top_3_2_contours[3]
    assignment_part12_left = (top_3_2_contours[2] + top_3_2_contours[0]) // 2
    coords_dict['assignment_part12'] = [int(assignment_part12_left), int(assignment_part12_top)]

    assignment_part13_top = top_3_3_contours[3]
    assignment_part13_left = (top_3_3_contours[0] + top_3_3_contours[2]) // 2
    coords_dict['assignment_part13'] = [int(assignment_part13_left), int(assignment_part13_top)]

    assignment_part14_top = top_3_4_contours[3]
    assignment_part14_left = (top_4_4_contours[0] + top_4_4_contours[2]) // 2
    coords_dict['assignment_part14'] = [int(assignment_part14_left), int(assignment_part14_top)]

    assignment_part21_top = top_4_1_contours[3]
    assignment_part21_left = (top_4_1_contours[2] + top_4_1_contours[0]) // 2
    coords_dict['assignment_part21'] = [int(assignment_part21_left), int(assignment_part21_top)]

    assignment_part22_top = top_4_2_contours[3]
    assignment_part22_left = (top_4_2_contours[2] + top_4_2_contours[0]) // 2
    coords_dict['assignment_part22'] = [int(assignment_part22_left), int(assignment_part22_top)]

    assignment_part23_top = top_4_3_contours[3]
    assignment_part23_left = (top_4_3_contours[2] + top_4_3_contours[0]) // 2
    coords_dict['assignment_part23'] = [int(assignment_part23_left), int(assignment_part23_top)]

    assignment_part24_top = top_4_4_contours[3]
    assignment_part24_left = (top_4_4_contours[2] + top_4_4_contours[0]) // 2
    coords_dict['assignment_part24'] = [int(assignment_part24_left), int(assignment_part24_top)]

    assignment_part31_top = top_5_1_contours[3]
    assignment_part31_left = (top_5_1_contours[2] + top_5_1_contours[0]) // 2
    coords_dict['assignment_part31'] = [int(assignment_part31_left), int(assignment_part31_top)]

    assignment_part32_top = top_5_2_contours[3]
    assignment_part32_left = (top_5_2_contours[2] + top_5_2_contours[0]) // 2
    coords_dict['assignment_part32'] = [int(assignment_part32_left), int(assignment_part32_top)]

    assignment_part33_top = top_5_4_contours[3]
    assignment_part33_left = (top_5_4_contours[2] + top_5_4_contours[0]) // 2
    coords_dict['assignment_part33'] = [int(assignment_part33_left), int(assignment_part33_top)]

    assignment_part34_top = top_5_5_contours[3]
    assignment_part34_left = (top_5_5_contours[2] + top_5_5_contours[0]) // 2
    coords_dict['assignment_part34'] = [int(assignment_part34_left), int(assignment_part34_top)]

    assignment_part35_top = top_5_6_contours[3]
    assignment_part35_left = (top_5_6_contours[2] + top_5_6_contours[0]) // 2
    coords_dict['assignment_part35'] = [int(assignment_part35_left), int(assignment_part35_top)]

    assignment_part36_top = top_5_8_contours[3]
    assignment_part36_left = (top_5_8_contours[2] + top_5_8_contours[0]) // 2
    coords_dict['assignment_part36'] = [int(assignment_part36_left), int(assignment_part36_top)]

    SBD_part1 = image[SBD_part1_top:top_2_right_contours[1], SBD_part1_left:(top_2_right_contours[0] + 50)]
    SBD_part2 = image[SBD_part2_top:top_2_right_contours[1], SBD_part2_left:(top_3_4_contours[0] + 50)]
    assignment_part11 = image[assignment_part11_top:top_4_2_contours[1], assignment_part11_left:((top_4_2_contours[0] + top_4_2_contours[2]) // 2)]
    assignment_part12 = image[assignment_part12_top:top_4_3_contours[1], assignment_part12_left:((top_4_3_contours[0] + top_4_3_contours[2]) // 2)]
    assignment_part13 = image[assignment_part13_top:top_4_4_contours[1], assignment_part13_left:((top_4_4_contours[0] + top_4_4_contours[2]) // 2)]
    assignment_part14 = image[assignment_part14_top:top_4_5_contours[1], assignment_part14_left:((top_4_5_contours[0] + top_4_5_contours[2]) // 2)]
    assignment_part21 = image[assignment_part21_top:top_5_3_contours[1], assignment_part21_left:((top_5_3_contours[0] + top_5_3_contours[2]) // 2)]
    assignment_part22 = image[assignment_part22_top:top_5_5_contours[1], assignment_part22_left:((top_5_5_contours[0] + top_5_5_contours[2]) // 2)]
    assignment_part23 = image[assignment_part23_top:top_5_7_contours[1], assignment_part23_left:((top_5_7_contours[0] + top_5_7_contours[2]) // 2)]
    assignment_part24 = image[assignment_part24_top:top_5_9_contours[1], assignment_part24_left:((top_5_9_contours[0] + top_5_9_contours[2]) // 2)]
    assignment_part31 = image[assignment_part31_top:top_6_1_contours[1], assignment_part31_left:((top_6_1_contours[0] + top_6_1_contours[2]) // 2)]
    assignment_part32 = image[assignment_part32_top:top_6_2_contours[1], assignment_part32_left:((top_6_2_contours[0] + top_6_2_contours[2]) // 2)]
    assignment_part33 = image[assignment_part33_top:top_6_3_contours[1], assignment_part33_left:((top_6_3_contours[0] + top_6_3_contours[2]) // 2)]
    assignment_part34 = image[assignment_part34_top:top_6_4_contours[1], assignment_part34_left:((top_6_4_contours[0] + top_6_4_contours[2]) // 2)]
    assignment_part35 = image[assignment_part35_top:top_6_5_contours[1], assignment_part35_left:((top_6_5_contours[0] + top_6_5_contours[2]) // 2)]
    assignment_part36 = image[assignment_part36_top:bot_right_contours[1], assignment_part36_left:((top_5_9_contours[0] + top_5_9_contours[2]) // 2)]

    return SBD_part1, SBD_part2, assignment_part11, assignment_part12, assignment_part13, assignment_part14, assignment_part21, assignment_part22, assignment_part23, assignment_part24, assignment_part31, assignment_part32, assignment_part33, assignment_part34, assignment_part35, assignment_part36


def process_info_part(array_info):
    one_positions = {}
    flag = [True] * 9 
    for col in range(array_info.shape[1]):
        rows_with_one = np.where(array_info[:, col] == 1)[0]
        if rows_with_one.size > 0:
            one_positions[col] = rows_with_one.tolist()
            flag[col] = len(rows_with_one) == 1

    SBD = "".join(str(one_positions[i][0]) for i in range(6) if i in one_positions)
    MaDe = "".join(str(one_positions[i][0]) for i in range(6, 9) if i in one_positions)

    return SBD, MaDe


def process_assignment_part1(arr_part1):
    result = []
    for i, row in enumerate(arr_part1):
        ones_count = np.sum(row)
        if ones_count == 1:
            if row[0] == 1:
                answer = 'A'
            elif row[1] == 1:
                answer = 'B'
            elif row[2] == 1:
                answer = 'C'
            elif row[3] == 1:
                answer = 'D'
        else:
            answer = '_'
        result.append(answer)
    
    return result


def process_assignment_part2(arr_part2):
    result = []
    for i in range(8):
        answer_string = ""
        for row in range(4):
            first_col = arr_part2[row, 2 * i]    
            second_col = arr_part2[row, 2 * i + 1] 
        
            if first_col == 1 and second_col == 0:
                answer_string += "D"
            elif first_col == 0 and second_col == 1:
                answer_string += "S"
            else:
                answer_string += "_"  
        result.append(answer_string)

    return result


def process_part3(matrix):
    result_string = ""
    decimal_added = False  

    if matrix[0, 0] == 1:
        result_string += "-"

    for col in range(matrix.shape[1]):  
        for row in range(2, matrix.shape[0]): 
            if matrix[row, col] == 1:
                result_string += str(row - 2)  

                if not decimal_added and (matrix[1, 1] == 1 or matrix[1, 2] == 1):
                    result_string += "."
                    decimal_added = True

    return result_string


def read_result(path):
    rows = []
    # Đọc file theo dòng, rồi tách thủ công làm 3 phần:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # split tối đa 2 lần: part, question_id, answer_id
            splitted = line.strip().split(',', 2)
            part = int(splitted[0])
            question_id = int(splitted[1])
            answer_id = splitted[2]
            rows.append({
                'part': part,
                'question_id': question_id,
                'answer_id': answer_id
            })
    
    # Tạo DataFrame
    true_data = pd.DataFrame(rows)

    # Tạo một bản copy để hiển thị nếu cần
    true_data_for_display = true_data.copy()

    # Map dùng để xử lý dữ liệu
    map_dict = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3,
        'Đúng': 0, 'Sai': 1,
        '-': 0, '.': 1, '0': 2, '1': 3, '2': 4, '3': 5,
        '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11,
        'Chưa chọn': 12
    }

    for i in range(len(true_data)):
        if true_data.loc[i, "part"] == 1 or true_data.loc[i, "part"] == 2:
            # Ánh xạ đáp án đơn (A/B/C/D hoặc Đúng/Sai => 0/1/2/3...)
            true_data.loc[i, "answer_id"] = map_dict[true_data.loc[i, "answer_id"]]
        else:
            # Xử lý part = 3
            answer_str = str(true_data.loc[i, "answer_id"]).strip()
            
            # Trường hợp "Chưa chọn" => gán ngay mã 12
            if answer_str == "Chưa chọn":
                true_data.loc[i, "answer_id"] = 12
            else:
                # Chuyển từ chuỗi "[12, 2, 3, 4]" thành list [12, 2, 3, 4]
                # nếu dữ liệu đúng format
                if answer_str.startswith("[") and answer_str.endswith("]"):
                    # parse thành list Python
                    answer_list = ast.literal_eval(answer_str)  # [12, 2, 3, 4]
                    
                    # Nếu bạn cần tiếp tục map từng phần tử
                    # (trường hợp mỗi phần tử là chuỗi '0','1','2'... thì phải ép sang str)
                    list_convert = []
                    for item in answer_list:
                        # Ví dụ: nếu item là số nguyên 2, ta cần dùng map_dict["2"] => 4
                        # Hoặc nếu item đã đúng thì cứ giữ nguyên
                        if str(item) in map_dict:
                            list_convert.append(map_dict[str(item)])
                        else:
                            # nếu không có trong map_dict, ta cho vào trực tiếp
                            list_convert.append(item)
                    
                    # Nếu list < 4 phần tử thì đệm thêm 12 cho đủ
                    if len(list_convert) < 4:
                        list_convert.extend([12] * (4 - len(list_convert)))
                    
                    # Gán lại vào DataFrame (ở dạng chuỗi cho đồng bộ)
                    true_data.loc[i, "answer_id"] = str(list_convert)
                
                else:
                    # Còn nếu dữ liệu không ở dạng list (ví dụ "2" hoặc "4"),
                    # ta vẫn dùng logic cũ (duyệt từng ký tự) hoặc tùy biến.
                    list_convert = []
                    for char in answer_str:
                        list_convert.append(map_dict[char])
                    if len(list_convert) < 4:
                        list_convert.extend([12] * (4 - len(list_convert)))
                    
                    true_data.loc[i, "answer_id"] = str(list_convert)

    return true_data, true_data_for_display

def compare_result(true_data, check_data):
  score_1 = 0
  score_2 = 0
  score_3 = 0
  MAX_QUESTION_PART_1 = 1e-10
  MAX_QUESTION_PART_2 = 1e-10
  MAX_QUESTION_PART_3 = 1e-10
  for i in range(len(true_data)):
    if int(true_data.iloc[i]["part"]) == 1:
      if true_data.iloc[i]["answer_id"] != 12:
        MAX_QUESTION_PART_1 += 1
        if int(true_data.iloc[i]["answer_id"]) == int(check_data.iloc[i]["answer_id"]):
          score_1 += 1
    elif int(true_data.iloc[i]["part"]) == 3:
      if true_data.iloc[i]["answer_id"] != 12:
        MAX_QUESTION_PART_3 += 1
        flag = True
        true_string = true_data.iloc[i]["answer_id"]
        true_string = true_string.replace("[", "")
        true_string = true_string.replace("]", "")
        true_array = [int(x) for x in true_string.split(",")]
        for j in range(4):
          if int(true_array[j]) != int(check_data.iloc[i]["answer_id"][j]):
            flag = False
        if flag:
          score_3 += 1


  all_answer_question_2 = true_data[true_data["part"] == 2]
  for i in range(8):
    score_check = 0
    check_nan = 0
    for j1 in range(i*4, (i+1)*4):
      if true_data.iloc[j1]["answer_id"] == 12:
        check_nan += 1
    if check_nan != 4:
      MAX_QUESTION_PART_2 += 1
      for j in range(i*4, (i+1)*4):
        if int(true_data.iloc[j]["answer_id"]) == int(all_answer_question_2.iloc[j]["answer_id"]):
          score_check += 1
      if score_check == 4:
        score_2 += 1
  
  return score_1, score_2, score_3, MAX_QUESTION_PART_1, MAX_QUESTION_PART_2, MAX_QUESTION_PART_3

def convert_to_final_score(score_1, score_2, score_3, MAX_PART_1, MAX_PART_2, MAX_PART_3, WEIGHT_PART_1, WEIGHT_PART_2, WEIGHT_PART_3):
  final_score = (score_1/MAX_PART_1)*WEIGHT_PART_1 + (score_2/MAX_PART_2)*WEIGHT_PART_2 + (score_3/MAX_PART_3)*WEIGHT_PART_3
  return round(final_score, 2)

def preprocess(src):
  median = cv2.medianBlur(src, 15)


  # Find local maximum
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 70))
  localmax = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel, iterations=1, borderType=cv2.BORDER_REFLECT101)

  # Compute the per pixel gain in a vectorized manner for all three channels
  gain = 255.0 / localmax.astype(np.float32)  # Ensure float division to get correct gain
  dst = np.clip(gain * src.astype(np.float32), 0, 255).astype(np.uint8)

  cv2.imwrite('output_image.jpg', dst)

  img = cv2.imread("output_image.jpg")
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 181, 3)
  
  kernel = np.ones((4, 4), np.uint8)
  thresh = cv2.dilate(thresh, kernel, iterations=1)
  # Avoid division by zero by setting zero values in thresh to a small positive number
  safe_thresh = np.where(thresh == 0, 1e-10, thresh)

  # Repeat thresh to match the shape of dst if dst has multiple channels
  if dst.ndim == 3 and thresh.ndim == 2:
      safe_thresh = np.repeat(safe_thresh[:, :, np.newaxis], dst.shape[2], axis=2)

  # Compute new_dst
  new_dst = 100 / safe_thresh * dst

  # Ensure new_dst is within valid image range
  new_dst = np.clip(new_dst, 0, 255).astype(np.uint8)
  cv2.imwrite('output_image.jpg', new_dst)
  return new_dst


if __name__ == "__main__":
    img_path = "../exams/LTHDT_all/0005.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    SBD_part1, SBD_part2, assignment_part11, assignment_part12, assignment_part13, assignment_part14, assignment_part21, assignment_part22, assignment_part23, assignment_part24, assignment_part31, assignment_part32, assignment_part33, assignment_part34, assignment_part35, assignment_part36 = split_image(img)

    cv2.imwrite("SBD_part1.png", SBD_part1)
    cv2.imwrite("SBD_part2.png", SBD_part2)


