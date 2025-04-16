import cv2
import numpy as np
import os
import random

def alignImages(scan_img, ref_img_gray, target_size=(2480, 3508)):
    """
    Aligns the scanned image to the reference image using SIFT and FLANN.

    Args:
        scan_img: The scanned image to be aligned.
        ref_img_gray: The reference image in grayscale.
        target_size: The target size for the aligned image.

    Returns:
        return aligned_img: The aligned image.
    """
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
        for m, n in knn_matches:
            if m.distance < threshold * n.distance:
                good_matches.add(m)
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


def findSquares(image_gray, template_gray, threshold):
    """
    Finds squares in the image using template matching and non-maximum suppression.

    Args:
        image_gray: The grayscale image to search for squares.
        template_gray: The grayscale template image.
        threshold: The threshold for template matching.

    Returns:
        squares: A list of detected squares with their coordinates and sizes.
    """
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
        temp_detections = []
        for x2, y2, _, _, _ in detections:
            if not (x1 < x2 + w and x1 + w > x2 and y1 < y2 + h and y1 + h > y2):
                temp_detections.append((x2, y2, w, h, confidence))
        detections = temp_detections
    return squares


# Function to find large squares and small squares using different thresholds
def findAllSquares(image):
    """
    Function to find all squares in the image using different thresholds.

    Args:
        image: The input image to find squares in.

    Returns:
        large_squares: List of detected large squares.
        small_squares: List of detected small squares.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for large_thresh in np.arange(0.6, 0.95, 0.05):
        large_squares = findSquares(image_gray, large_square_template_gray, large_thresh)
        if len(large_squares) == 4:
            print(f"Found 4 large squares with threshold {large_thresh}")
            break
    if len(large_squares) != 4:
        raise ValueError("Could not find the required number of large squares after trying different thresholds.")

    for small_thresh in np.arange(0.6, 0.99, 0.01):
        small_squares = findSquares(image_gray, small_square_template_gray, small_thresh)
        if len(small_squares) == 27:
            print(f"Found 27 small squares with threshold {small_thresh}")
            break
    if len(small_squares) != 27:
        print(f"Len of small_squares: {len(small_squares)}")
        raise ValueError("Could not find the required number of squares after trying different thresholds.")
    
    large_squares = sorted(large_squares, key=lambda x: (x[1] + x[0]))
    small_squares = sorted(small_squares, key=lambda x: (x[1] + x[0]))


    print("[DEBUG]\n Large squares found: ", len(large_squares))
    for idx, (x, y, w, h) in enumerate(large_squares):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, str(idx), (x + w + 10, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    print("Small squares found: ", len(small_squares))
    for idx, (x, y, w, h) in enumerate(small_squares):
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, str(idx), (x + w + 10, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite("result.png", image)
    print("Export result.png\n")
    return large_squares, small_squares


def padding_image(image):
    """
    Function to pad the image to a square size of 640x640 pixels.

    Args:
        image: The input image to be padded.

    Returns:
        padded_image: The padded image.
    """
    h, w = image.shape[:2]
    scale = 640 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    image = cv2.resize(image, (new_w, new_h))

    pad_w, pad_h = 640 - new_w, 640 - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


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

        # # Metadata
        # studentID_c1x = large_squares[0][0] + large_squares[0][2]
        # studentID_c1y = small_squares[0][1] + small_squares[0][3] // 2
        # studentID_c2x = small_squares[3][0] 
        # studentID_c2y = small_squares[3][1]
        # print(f"StudentID c1: {studentID_c1x}, {studentID_c1y}, c2: {studentID_c2x}, {studentID_c2y}")

        # examID_c1x = small_squares[0][0] + small_squares[0][2]
        # examID_c1y = small_squares[0][1] + small_squares[0][3] // 2
        # examID_c2x = small_squares[6][0]
        # examID_c2y = small_squares[6][1]
        # print(f"ExamID c1: {examID_c1x}, {examID_c1y}, c2: {examID_c2x}, {examID_c2y}")

        # # Content
        # content11_c1x = small_squares[2][0] + int(1.8 * small_squares[2  ][2])
        # content11_c1y = small_squares[2][1] + small_squares[2][3] + 5
        # content11_c2x = small_squares[8][0]
        # content11_c2y = small_squares[8][1] - small_squares[8][3]   
        # print(f"Content11 c1: {content11_c1x}, {content11_c1y}, c2: {content11_c2x}, {content11_c2y}")
        
        # content12_c1x = small_squares[5][0] + small_squares[5][2]
        # content12_c1y = small_squares[5][1] + small_squares[5][3]
        # content12_c2x = small_squares[10][0]
        # content12_c2y = small_squares[10][1] - small_squares[10][3]
        # print(f"Content12 c1: {content12_c1x}, {content12_c1y}, c2: {content12_c2x}, {content12_c2y}")

        # content13_c1x = small_squares[9][0] + small_squares[9][2]
        # content13_c1y = small_squares[9][1] + small_squares[9][3]
        # content13_c2x = small_squares[13][0]
        # content13_c2y = small_squares[13][1] - small_squares[13][3]
        # print(f"Content13 c1: {content13_c1x}, {content13_c1y}, c2: {content13_c2x}, {content13_c2y}")

        # content14_c1x = small_squares[12][0] + small_squares[12][2]
        # content14_c1y = small_squares[12][1] + small_squares[12][3]
        # content14_c2x = small_squares[16][0]
        # content14_c2y = small_squares[16][1] - small_squares[16][3]
        # print(f"Content14 c1: {content14_c1x}, {content14_c1y}, c2: {content14_c2x}, {content14_c2y}")

        # content21_c1x = small_squares[4][0] + int(1.8 * small_squares[4][2])
        # content21_c1y = small_squares[4][1] + small_squares[4][3]
        # content21_c2x = small_squares[11][0]
        # content21_c2y = small_squares[11][1] 
        # print(f"Content21 c1: {content21_c1x}, {content21_c1y}, c2: {content21_c2x}, {content21_c2y}")

        # content22_c1x = small_squares[8][0] + small_squares[8][2]
        # content22_c1y = small_squares[8][1] + small_squares[8][3]
        # content22_c2x = small_squares[14][0]
        # content22_c2y = small_squares[14][1]    
        # print(f"Content22 c1: {content22_c1x}, {content22_c1y}, c2: {content22_c2x}, {content22_c2y}")

        # content23_c1x = small_squares[10][0] + small_squares[10][2]
        # content23_c1y = small_squares[10][1] + small_squares[10][3]
        # content23_c2x = small_squares[17][0]
        # content23_c2y = small_squares[17][1] 
        # print(f"Content23 c1: {content23_c1x}, {content23_c1y}, c2: {content23_c2x}, {content23_c2y}")

        # content24_c1x = small_squares[13][0] + small_squares[13][2]
        # content24_c1y = small_squares[13][1] + small_squares[13][3]
        # content24_c2x = small_squares[18][0]
        # content24_c2y = small_squares[18][1]
        # print(f"Content24 c1: {content24_c1x}, {content24_c1y}, c2: {content24_c2x}, {content24_c2y}")

        # # Split image
        # studentID = align_image[studentID_c1y:studentID_c2y, studentID_c1x: studentID_c2x]
        # examID = align_image[examID_c1y:examID_c2y, examID_c1x: examID_c2x]

        # content11 = align_image[content11_c1y:content11_c2y, content11_c1x: content11_c2x]
        # content12 = align_image[content12_c1y:content12_c2y, content12_c1x: content12_c2x]
        # content13 = align_image[content13_c1y:content13_c2y, content13_c1x: content13_c2x]
        # content14 = align_image[content14_c1y:content14_c2y, content14_c1x: content14_c2x]
        # content21 = align_image[content21_c1y:content21_c2y, content21_c1x: content21_c2x]
        # content22 = align_image[content22_c1y:content22_c2y, content22_c1x: content22_c2x]
        # content23 = align_image[content23_c1y:content23_c2y, content23_c1x: content23_c2x]
        # content24 = align_image[content24_c1y:content24_c2y, content24_c1x: content24_c2x]

        # # Padding split images
        # studentID = padding_image(studentID)
        # examID = padding_image(examID)
        # content11 = padding_image(content11)
        # content12 = padding_image(content12)
        # content13 = padding_image(content13)
        # content14 = padding_image(content14)
        # content21 = padding_image(content21)
        # content22 = padding_image(content22)
        # content23 = padding_image(content23)
        # content24 = padding_image(content24)

        # # Save images
        # cv2.imwrite(unlabel_metadata_folder_path + "/" + image_base_name + "_studentID.png", studentID)
        # cv2.imwrite(unlabel_metadata_folder_path + "/" + image_base_name + "_examID.png", examID)
        # cv2.imwrite(unlabel_content_folder_path + "/" + image_base_name + "_content11.png", content11)
        # cv2.imwrite(unlabel_content_folder_path + "/" + image_base_name + "_content12.png", content12)
        # cv2.imwrite(unlabel_content_folder_path + "/" + image_base_name + "_content13.png", content13)
        # cv2.imwrite(unlabel_content_folder_path + "/" + image_base_name + "_content14.png", content14)
        # cv2.imwrite(unlabel_content_folder_path + "/" + image_base_name + "_content21.png", content21)
        # cv2.imwrite(unlabel_content_folder_path + "/" + image_base_name + "_content22.png", content22)
        # cv2.imwrite(unlabel_content_folder_path + "/" + image_base_name + "_content23.png", content23)
        # cv2.imwrite(unlabel_content_folder_path + "/" + image_base_name + "_content24.png", content24)    
    
    except Exception as e:
        print(f"Error: {e}")    


# Main function
if __name__ == '__main__':
    # Read reference image and templates
    ref_img_path = "../../ref_tmpl/QM-2025.jpg" 
    ref_img_gray = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    ref_img_gray = cv2.resize(ref_img_gray, (2480, 3508))

    square_template_path = "../../ref_tmpl/template.png"
    square_template_gray = cv2.imread(square_template_path, cv2.IMREAD_GRAYSCALE)
    large_square_template_gray = cv2.resize(square_template_gray, (80, 80))
    small_square_template_gray = cv2.resize(square_template_gray, (40, 40))

    # Read folders
    scan_images_folder_path = "../../old_data/scan_images"

    align_images_folder_path = "../../old_data/align_images"
    if not os.path.exists(align_images_folder_path):
        os.makedirs(align_images_folder_path) 

    unlabel_metadata_folder_path = "../../old_data/unlabel_metadata"
    if not os.path.exists(unlabel_metadata_folder_path):
        os.makedirs(unlabel_metadata_folder_path) 

    unlabel_content_folder_path = "../../old_data/unlabel_content"
    if not os.path.exists(unlabel_content_folder_path):
        os.makedirs(unlabel_content_folder_path) 

    # Split images
    for image_file in os.listdir(scan_images_folder_path):
        image_path = os.path.join(scan_images_folder_path, image_file)
        image = cv2.imread(image_path)  
        image = cv2.resize(image, (2480, 3508))
        splitImage(image)
        print(f"Image {image_file} split successfully.")
        print("=====================================\n")

