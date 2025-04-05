import cv2
import numpy as np
import os
import random


# Function to match keypoints using Lowe's ratio test
def loweMatch(knn_matches, good_matches, threshold):
    for m, n in knn_matches:
        if m.distance < threshold * n.distance:
            good_matches.add(m)
    return good_matches


# Function to align images using SIFT and FLANN
def alignImages(scan_img, ref_img_gray, target_size=(2480, 3508)):
    scan_img_gray = cv2.cvtColor(scan_img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()
    scan_keypoints, scan_descriptors = sift.detectAndCompute(scan_img_gray, None)
    ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_img_gray, None)

    if scan_descriptors is None or ref_descriptors is None:
        raise ValueError("Failed to compute SIFT descriptors.")

    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches using k-NN and Lowe ratio
    knn_matches = flann.knnMatch(scan_descriptors, ref_descriptors, k=2)    
    good_matches = set()
    threshold = 0.5
    while len(good_matches) < 500:
        good_matches = loweMatch(knn_matches, good_matches, threshold)
        if threshold < 1:
            threshold += 0.1
        else:
            raise ValueError("Can not find at least 500 good matches.")
    print(f"Good matches found: {len(good_matches)} with threshold {threshold - 0.1}")

    # Extract matched keypoints locations
    points1 = np.float32([scan_keypoints[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([ref_keypoints[m.trainIdx].pt for m in good_matches])

    # Compute homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    if h is None:
        raise ValueError("Homography estimation failed.")

    # Align
    aligned_img = cv2.warpPerspective(scan_img, h, target_size, flags=cv2.INTER_LINEAR)
    
    return aligned_img


# Function to find squares using template matching and NMS
def findSquares(image_gray, template_gray, threshold):
    # Find squares
    h, w = template_gray.shape
    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # NMS
    detections = []
    for pt in zip(*loc[::-1]):  
        detections.append((pt[0], pt[1], w, h, res[pt[1], pt[0]]))
    detections.sort(key=lambda x: x[4], reverse=True)  

    squares = []
    while detections:
        x1, y1, w, h, confidence = detections.pop(0)
        squares.append((x1, y1, w, h))
        # Remove overlapping detections
        temp_detections = []
        for x2, y2, _, _, _ in detections:
            if not (x1 < x2 + w and x1 + w > x2 and y1 < y2 + h and y1 + h > y2):
                temp_detections.append((x2, y2, w, h, confidence))
        detections = temp_detections

    # print("[DEBUG] Large squares found: ", len(large_squares))
    # for idx, (x, y, w, h) in enumerate(large_squares):
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.putText(image, str(idx), (x + w + 10, y + h - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # cv2.imwrite("result.png", image)
    # print("[DEBUG] Export result.png"))

    return squares


# Function to find large squares and small squares using different thresholds
def findAllSquares(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for large_thresh in np.arange(0.6, 0.95, 0.05):
        large_squares = findSquares(image_gray, large_square_template_gray, large_thresh)
        if len(large_squares) == 4:
            print(f"Found 4 large squares with threshold {large_thresh}")
            break
    if len(large_squares) != 4:
        raise ValueError("Could not find the required number of large squares after trying different thresholds.")

    for small_thresh in np.arange(0.7, 0.95, 0.01):
        small_squares = findSquares(image_gray, small_square_template_gray, small_thresh)
        if len(small_squares) == 19:
            print(f"Found 19 small squares with threshold {small_thresh}")
            break
    if len(small_squares) != 19:
        print(f"Len of small_squares: {len(small_squares)}")
        raise ValueError("Could not find the required number of squares after trying different thresholds.")
    
    large_squares = sorted(large_squares, key=lambda x: (x[1] + x[0]))
    small_squares = sorted(small_squares, key=lambda x: (x[1] + x[0]))

    return large_squares, small_squares


# Function to create image versions with conditional padding
def createImageVersions(rotate=4, move=True, image=None, target_size=640, padding_color=[255, 255, 255]):
    output_images = []
    original_image = image.copy()

    # Define rotate degree
    if rotate == 4:
        rotation_codes = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        ]
    else:
        rotation_codes = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
        ]

    for rot_code in rotation_codes:
        # Rotate
        if rot_code is None:
            rotated_image = original_image
        else:
            rotated_image = cv2.rotate(original_image, rot_code)

        # Scale
        h, w = rotated_image.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(rotated_image, (new_w, new_h))

        total_pad_w = max(0, target_size - new_w)
        total_pad_h = max(0, target_size - new_h)

        # Padding to create 3 versions
        # --- Center version  ---
        left_c = total_pad_w // 2
        right_c = total_pad_w - left_c
        top_c = total_pad_h // 2
        bottom_c = total_pad_h - top_c
        center_padded = cv2.copyMakeBorder(resized_image, top_c, bottom_c, left_c, right_c,
                                           cv2.BORDER_CONSTANT, value=padding_color)
        output_images.append(center_padded)

         # --- Conditional padding ---
        if move:
            if new_w >= new_h: # Wider or square image -> Pad Top/Center/Bottom
                left_tb = left_c
                right_tb = right_c

                # Version B: Top-biased padding
                if total_pad_h > 0:
                    max_top_pad = max(0, total_pad_h // 4)
                    top_t = random.randint(0, max_top_pad)
                    bottom_t = total_pad_h - top_t
                else:
                    top_t, bottom_t = 0, 0
                top_padded = cv2.copyMakeBorder(resized_image, top_t, bottom_t, left_tb, right_tb,
                                                cv2.BORDER_CONSTANT, value=padding_color)
                output_images.append(top_padded)

                # Version C: Bottom-biased padding
                if total_pad_h > 0:
                    max_bottom_pad = max(0, total_pad_h // 4)
                    bottom_b = random.randint(0, max_bottom_pad)
                    top_b = total_pad_h - bottom_b
                else:
                    top_b, bottom_b = 0, 0
                bottom_padded = cv2.copyMakeBorder(resized_image, top_b, bottom_b, left_tb, right_tb,
                                                cv2.BORDER_CONSTANT, value=padding_color)
                output_images.append(bottom_padded)

            else: # Taller image -> Pad Left/Center/Right
                top_lr = top_c
                bottom_lr = bottom_c

                # Version B: Left-biased padding
                if total_pad_w > 0:
                    max_left_pad = max(0, total_pad_w // 3)
                    left_l = random.randint(0, max_left_pad)
                    right_l = total_pad_w - left_l
                else:
                    left_l, right_l = 0, 0
                left_padded = cv2.copyMakeBorder(resized_image, top_lr, bottom_lr, left_l, right_l,
                                                cv2.BORDER_CONSTANT, value=padding_color)
                output_images.append(left_padded)

                # Version C: Right-biased padding
                if total_pad_w > 0:
                    max_right_pad = max(0, total_pad_w // 3)
                    right_r = random.randint(0, max_right_pad)
                    left_r = total_pad_w - right_r
                else:
                    left_r, right_r = 0, 0
                right_padded = cv2.copyMakeBorder(resized_image, top_lr, bottom_lr, left_r, right_r,
                                                cv2.BORDER_CONSTANT, value=padding_color)
                output_images.append(right_padded)

    return output_images


# Pipeline to split image
def splitImage(image):
    try:
        # Align image
        align_image = alignImages(image, ref_img_gray)
        image_base_name = image_file.split(".")[0]
        align_image_name = image_base_name + "_align.png"
        align_image_path = align_images_folder_path + "/" + align_image_name
        cv2.imwrite(align_image_path, align_image)

        # Assign coordinate
        # c1 = top_left_corner, c2 = bottom_right_corner
        large_squares, small_squares = findAllSquares(align_image)

        # ID part
        student_id_c1x = large_squares[0][0] + large_squares[0][2]
        student_id_c1y = small_squares[0][1] + small_squares[0][3] // 2
        student_id_c2x = small_squares[3][0] 
        student_id_c2y = small_squares[3][1]

        exam_id_c1x = small_squares[0][0] + small_squares[0][2]
        exam_id_c1y = small_squares[0][1] + small_squares[0][3] // 2
        exam_id_c2x = small_squares[6][0]
        exam_id_c2y = small_squares[6][1]

        # Assignment
        asgm_part11_c1x = small_squares[2][0] + int(1.8 * small_squares[2][2])
        asgm_part11_c1y = small_squares[2][1] + small_squares[2][3] + 5
        asgm_part11_c2x = small_squares[8][0]
        asgm_part11_c2y = small_squares[8][1] - small_squares[8][3]
        
        asgm_part12_c1x = small_squares[5][0] + small_squares[5][2]
        asgm_part12_c1y = small_squares[5][1] + small_squares[5][3]
        asgm_part12_c2x = small_squares[10][0]
        asgm_part12_c2y = small_squares[10][1] - small_squares[10][3]

        asgm_part13_c1x = small_squares[9][0] + small_squares[9][2]
        asgm_part13_c1y = small_squares[9][1] + small_squares[9][3]
        asgm_part13_c2x = small_squares[13][0]
        asgm_part13_c2y = small_squares[13][1] - small_squares[13][3]

        asgm_part14_c1x = small_squares[12][0] + small_squares[12][2]
        asgm_part14_c1y = small_squares[12][1] + small_squares[12][3]
        asgm_part14_c2x = small_squares[16][0]
        asgm_part14_c2y = small_squares[16][1] - small_squares[16][3]

        asgm_part21_c1x = small_squares[4][0] + int(1.8 * small_squares[4][2])
        asgm_part21_c1y = small_squares[4][1] + small_squares[4][3]
        asgm_part21_c2x = small_squares[11][0]
        asgm_part21_c2y = small_squares[11][1] 

        asgm_part22_c1x = small_squares[8][0] + small_squares[8][2]
        asgm_part22_c1y = small_squares[8][1] + small_squares[8][3]
        asgm_part22_c2x = small_squares[14][0]
        asgm_part22_c2y = small_squares[14][1] 

        asgm_part23_c1x = small_squares[10][0] + small_squares[10][2]
        asgm_part23_c1y = small_squares[10][1] + small_squares[10][3]
        asgm_part23_c2x = small_squares[17][0]
        asgm_part23_c2y = small_squares[17][1] 

        asgm_part24_c1x = small_squares[13][0] + small_squares[13][2]
        asgm_part24_c1y = small_squares[13][1] + small_squares[13][3]
        asgm_part24_c2x = small_squares[18][0]
        asgm_part24_c2y = small_squares[18][1]

        # Split image
        student_id = align_image[student_id_c1y:student_id_c2y, student_id_c1x: student_id_c2x]
        exam_id = align_image[exam_id_c1y:exam_id_c2y, exam_id_c1x: exam_id_c2x]

        asgm_part11 = align_image[asgm_part11_c1y:asgm_part11_c2y, asgm_part11_c1x: asgm_part11_c2x]
        asgm_part12 = align_image[asgm_part12_c1y:asgm_part12_c2y, asgm_part12_c1x: asgm_part12_c2x]
        asgm_part13 = align_image[asgm_part13_c1y:asgm_part13_c2y, asgm_part13_c1x: asgm_part13_c2x]
        asgm_part14 = align_image[asgm_part14_c1y:asgm_part14_c2y, asgm_part14_c1x: asgm_part14_c2x]
        asgm_part21 = align_image[asgm_part21_c1y:asgm_part21_c2y, asgm_part21_c1x: asgm_part21_c2x]
        asgm_part22 = align_image[asgm_part22_c1y:asgm_part22_c2y, asgm_part22_c1x: asgm_part22_c2x]
        asgm_part23 = align_image[asgm_part23_c1y:asgm_part23_c2y, asgm_part23_c1x: asgm_part23_c2x]
        asgm_part24 = align_image[asgm_part24_c1y:asgm_part24_c2y, asgm_part24_c1x: asgm_part24_c2x]

        # Padding split images
        student_id_images = createImageVersions(rotate=2, image=student_id)
        exam_id_images = createImageVersions(rotate=2, image=exam_id)
        asgm_part11_images = createImageVersions(image=asgm_part11)
        asgm_part12_images = createImageVersions(image=asgm_part12)
        asgm_part13_images = createImageVersions(image=asgm_part13)
        asgm_part14_images = createImageVersions(image=asgm_part14)
        asgm_part21_images = createImageVersions(move=False, image=asgm_part21)
        asgm_part22_images = createImageVersions(move=False, image=asgm_part22)
        asgm_part23_images = createImageVersions(move=False, image=asgm_part23)
        asgm_part24_images = createImageVersions(move=False, image=asgm_part24)


        # Save images
        for i in range(len(student_id_images)):
            student_id_path = unlabel_metadata_folder_path + "/" + image_base_name + "_student_id_v" + str(i) + ".png"
            cv2.imwrite(student_id_path, student_id_images[i])
        
        for i in range(len(exam_id_images)):
            exam_id_path = unlabel_metadata_folder_path + "/" + image_base_name + "_exam_id_v" + str(i) + ".png"
            cv2.imwrite(exam_id_path, exam_id_images[i])
        
        for i in range(len(asgm_part11_images)):        
            asgm_part11_path = unlabel_assignment_folder_path + "/" + image_base_name + "_asgm_part11_v" + str(i) + ".png"
            cv2.imwrite(asgm_part11_path, asgm_part11_images[i])

        for i in range(len(asgm_part12_images)):
            asgm_part12_path = unlabel_assignment_folder_path + "/" + image_base_name + "_asgm_part12_v" + str(i) + ".png"
            cv2.imwrite(asgm_part12_path, asgm_part12_images[i])

        for i in range(len(asgm_part13_images)):
            asgm_part13_path = unlabel_assignment_folder_path + "/" + image_base_name + "_asgm_part13_v" + str(i) + ".png"
            cv2.imwrite(asgm_part13_path, asgm_part13_images[i])

        for i in range(len(asgm_part14_images)):
            asgm_part14_path = unlabel_assignment_folder_path + "/" + image_base_name + "_asgm_part14_v" + str(i) + ".png"
            cv2.imwrite(asgm_part14_path, asgm_part14_images[i])

        for i in range(len(asgm_part21_images)):
            asgm_part21_path = unlabel_assignment_folder_path + "/" + image_base_name + "_asgm_part21_v" + str(i) + ".png"
            cv2.imwrite(asgm_part21_path, asgm_part21_images[i])

        for i in range(len(asgm_part22_images)):
            asgm_part22_path = unlabel_assignment_folder_path + "/" + image_base_name + "_asgm_part22_v" + str(i) + ".png"
            cv2.imwrite(asgm_part22_path, asgm_part22_images[i])

        for i in range(len(asgm_part23_images)):
            asgm_part23_path = unlabel_assignment_folder_path + "/" + image_base_name + "_asgm_part23_v" + str(i) + ".png"
            cv2.imwrite(asgm_part23_path, asgm_part23_images[i])

        for i in range(len(asgm_part24_images)):
            asgm_part24_path = unlabel_assignment_folder_path + "/" + image_base_name + "_asgm_part24_v" + str(i) + ".png"
            cv2.imwrite(asgm_part24_path, asgm_part24_images[i]) 
    
    except Exception as e:
        print(f"Error: {e}")    


# Main function
if __name__ == '__main__':
    # Read reference image and templates
    ref_img_path = "../ref_tmpl/reference.png" 
    ref_img_gray = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    ref_img_gray = cv2.resize(ref_img_gray, (2480, 3508))

    square_template_path = "../ref_tmpl/template.png"
    square_template_gray = cv2.imread(square_template_path, cv2.IMREAD_GRAYSCALE)
    large_square_template_gray = cv2.resize(square_template_gray, (80, 80))
    small_square_template_gray = cv2.resize(square_template_gray, (40, 40))

    # Read folders
    scan_images_folder_path = "../data/scan_images"

    align_images_folder_path = "../data/align_images"
    if not os.path.exists(align_images_folder_path):
        os.makedirs(align_images_folder_path) 

    unlabel_metadata_folder_path = "../data/unlabel_metadata"
    if not os.path.exists(unlabel_metadata_folder_path):
        os.makedirs(unlabel_metadata_folder_path) 

    unlabel_assignment_folder_path = "../data/unlabel_assignment"
    if not os.path.exists(unlabel_assignment_folder_path):
        os.makedirs(unlabel_assignment_folder_path) 

    # Split images
    for image_file in os.listdir(scan_images_folder_path):
        if image_file == "page_013.png":
            image_path = os.path.join(scan_images_folder_path, image_file)
            image = cv2.imread(image_path)  
            image = cv2.resize(image, (2480, 3508))
            splitImage(image)
            print(f"Image {image_file} split successfully.")
            print("=====================================\n")

