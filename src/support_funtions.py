import cv2
import numpy as np
import os
import random

# Read reference image
ref_img_path = "../reference/reference.png" 
ref_img_gray = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)


# Support functions
def loweMatch(knn_matches, good_matches, threshold):
    for m, n in knn_matches:
        if m.distance < threshold * n.distance:
            good_matches.add(m)
    return good_matches


def alignImages(scanned_img, ref_img_gray):
    scan_img_gray = cv2.cvtColor(scanned_img, cv2.COLOR_BGR2GRAY)

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
    threshold = 0.3
    while len(good_matches) < 4:
        good_matches = loweMatch(knn_matches, good_matches, threshold)
        if threshold < 1:
            threshold += 0.1
        else:
            raise ValueError("Can not find enough 4 good matches.")

    # Extract matched keypoints locations
    points1 = np.float32([scan_keypoints[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([ref_keypoints[m.trainIdx].pt for m in good_matches])

    # Compute homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    if h is None:
        raise ValueError("Homography estimation failed.")

    # Align
    aligned_img = cv2.warpPerspective(scanned_img, h, (2480, 3508))
    
    return aligned_img


def findSquares(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold (convert black to white)
    _, binary = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_squares = []
    small_squares = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4: 
            x, y, w, h = cv2.boundingRect(approx)

            if 60 <= w <= 80 and 60 <= h <= 80:
                large_squares.append((x, y, w, h))
            elif 24 <= w <= 44 and 24 <= h <= 44:
                small_squares.append((x, y, w, h))

    return large_squares, small_squares


def padding_image(image):
    h, w = image.shape[:2]
    scale = 640 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    image = cv2.resize(image, (new_w, new_h))

    pad_w, pad_h = 640 - new_w, 640 - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def create_image_versions(image, target_size=640, padding_color=[255, 255, 255]):
    output_images = []
    original_image = image.copy() 

    # Rtation codes 
    rotation_codes = [
        None,                        
        cv2.ROTATE_90_CLOCKWISE,      
        cv2.ROTATE_180,              
        cv2.ROTATE_90_COUNTERCLOCKWISE 
    ]

    for rot_code in rotation_codes:
        if rot_code is None:
            rotated_image = original_image
        else:
            rotated_image = cv2.rotate(original_image, rot_code)

        # Scale 
        h, w = rotated_image.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(rotated_image, (new_w, new_h))

        # 3. Calculate padding amounts
        total_pad_w = target_size - new_w
        total_pad_h = target_size - new_h

        # --- Create 3 padding versions ---
        # Version A: Centered padding
        left_c = total_pad_w // 2
        right_c = total_pad_w - left_c
        top_c = total_pad_h // 2
        bottom_c = total_pad_h - top_c
        center_padded = cv2.copyMakeBorder(resized_image, top_c, bottom_c, left_c, right_c,
                                           cv2.BORDER_CONSTANT, value=padding_color)
        output_images.append(center_padded)

        # Top bot take left right same as center
        left_tb = left_c
        right_tb = right_c

        # Version B: Top-Biased Padding 
        if total_pad_h > 0:
            max_top_pad = max(0, total_pad_h // 3)
            top_t = random.randint(0, max_top_pad)
            bottom_t = total_pad_h - top_t
        else: 
            top_t, bottom_t = 0, 0
        top_padded = cv2.copyMakeBorder(resized_image, top_t, bottom_t, left_tb, right_tb,
                                         cv2.BORDER_CONSTANT, value=padding_color)
        output_images.append(top_padded)

        # Version C: Bottom-Biased Padding 
        if total_pad_h > 0:
            # Randomly choose bottom padding amount, biased towards being small (e.g., 0 to 1/3rd of total)
            max_bottom_pad = max(0, total_pad_h // 3) # Ensure range starts from 0
            bottom_b = random.randint(0, max_bottom_pad)
            top_b = total_pad_h - bottom_b
        else: # No vertical padding needed
            top_b, bottom_b = 0, 0
        bottom_padded = cv2.copyMakeBorder(resized_image, top_b, bottom_b, left_tb, right_tb,
                                           cv2.BORDER_CONSTANT, value=padding_color)
        output_images.append(bottom_padded)

    return output_images

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy image for testing (e.g., 300x200, blue rectangle)
    # Replace this with: img = cv2.imread('your_image_path.jpg')
    test_image = np.zeros((200, 300, 3), dtype=np.uint8)
    test_image[:, :] = [255, 0, 0] # Blue color in BGR

    # Or load a real image
    # try:
    #     test_image = cv2.imread('path/to/your/image.png') # PUT YOUR IMAGE PATH HERE
    #     if test_image is None:
    #         raise FileNotFoundError("Image not found or could not be loaded.")
    # except FileNotFoundError as e:
    #     print(e)
    #     exit()
    # except Exception as e:
    #     print(f"An error occurred loading the image: {e}")
    #     exit()


    image_versions = create_image_versions(test_image, target_size=640)

    if image_versions:
        print(f"Successfully generated {len(image_versions)} versions.")

        # You can now process/save these versions
        # Example: Save the first 3 versions (0 degrees rotation)
        cv2.imwrite('output_0deg_center.png', image_versions[0])
        cv2.imwrite('output_0deg_top.png', image_versions[1])
        cv2.imwrite('output_0deg_bottom.png', image_versions[2])

        # Example: Save the next 3 versions (90 degrees rotation)
        cv2.imwrite('output_90deg_center.png', image_versions[3])
        cv2.imwrite('output_90deg_top.png', image_versions[4])
        cv2.imwrite('output_90deg_bottom.png', image_versions[5])
        # ... and so on for 180 and 270 degrees

        # Or display one
        # cv2.imshow('Example - 0 deg Center', image_versions[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Image version generation failed.")


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
        large_squares, small_squares = findSquares(align_image)
        large_squares.sort(key=lambda a: a[0] + a[1])
        small_squares.sort(key=lambda a: a[0] + a[1])

        # # Draw red boxes on large squares and label 
        # for idx, (x, y, w, h) in enumerate(large_squares):
        #     cv2.rectangle(align_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     cv2.putText(align_image, str(idx), (x + w + 10, y + h - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # # Draw blue boxes on small squares and label 
        # for idx, (x, y, w, h) in enumerate(small_squares):
        #     cv2.rectangle(align_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #     cv2.putText(align_image, str(idx), (x + w + 10, y + h - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # # Save
        # cv2.imwrite(align_image_name, align_image)

        # ID part
        student_id_c1x = large_squares[0][0] + large_squares[0][2]
        student_id_c1y = small_squares[0][1] + small_squares[0][3] // 2
        student_id_c2x = small_squares[3][0] 
        student_id_c2y = small_squares[3][1]

        exam_id_c1x = small_squares[0][0] + small_squares[0][2]
        exam_id_c1y = small_squares[0][1] + small_squares[0][3]
        exam_id_c2x = small_squares[6][0]
        exam_id_c2y = small_squares[6][1]

        # Assignment
        asgm_part11_c1x = small_squares[2][0] + int(1.8 * small_squares[2][2])
        asgm_part11_c1y = small_squares[2][1] + small_squares[2][3] + 10
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
        student_id = padding_image(student_id)
        exam_id = padding_image(exam_id)
        asgm_part11 = padding_image(asgm_part11)
        asgm_part12 = padding_image(asgm_part12)
        asgm_part13 = padding_image(asgm_part13)
        asgm_part14 = padding_image(asgm_part14)
        asgm_part21 = padding_image(asgm_part21)
        asgm_part22 = padding_image(asgm_part22)
        asgm_part23 = padding_image(asgm_part23)
        asgm_part24 = padding_image(asgm_part24)

        # Save images
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_student_id.png", student_id)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_exam_id.png", exam_id)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_asgm_part11.png", asgm_part11)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_asgm_part12.png", asgm_part12)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_asgm_part13.png", asgm_part13)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_asgm_part14.png", asgm_part14)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_asgm_part21.png", asgm_part21)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_asgm_part22.png", asgm_part22)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_asgm_part23.png", asgm_part23)
        cv2.imwrite(unlabel_data_folder_path + "/" + image_base_name + "_asgm_part24.png", asgm_part24)    
    
    except Exception as e:
        print(f"Unexpected error: {e}")    

if __name__ == '__main__':
    # Folders
    scan_images_folder_path = "../data/scan_images"

    align_images_folder_path = "../data/align_images"
    if not os.path.exists(align_images_folder_path):
        os.makedirs(align_images_folder_path) 

    unlabel_data_folder_path = "../data/unlabel_data"
    if not os.path.exists(unlabel_data_folder_path):
        os.makedirs(unlabel_data_folder_path) 

    # Split images
    for image_file in os.listdir(scan_images_folder_path):
        image_path = os.path.join(scan_images_folder_path, image_file)
        image = cv2.imread(image_path)  
        image = cv2.resize(image, (2480, 3508))
        splitImage(image)
        print(f"Image {image_file} split successfully.")

    print("=============================")
    print("Split images done!")





