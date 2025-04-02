from __future__ import print_function
import cv2
import numpy as np
import os


def loweMatch(knn_matches, good_matches, threshold=0.5):
    good_matches.clear()
    for m, n in knn_matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    
    return good_matches


def alignImages(ref_img, scanned_img, threshold=0.5):
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    scan_img_gray = cv2.cvtColor(scanned_img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(ref_img_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(scan_img_gray, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Failed to compute SIFT descriptors.")

    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches using k-NN and Lowe ratio
    knn_matches = flann.knnMatch(descriptors1, descriptors2, k=2)    
    good_matches = []
    threshold = 0.5

    while len(good_matches) < 4:
        good_matches = loweMatch(knn_matches, good_matches, threshold)
        if threshold < 0.95:
            threshold += 0.5
        else:
            raise ValueError("Can not find enough 4 good matches.")
        
    # Extract matched keypoints locations
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Compute homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    if h is None:
        raise ValueError("Homography estimation failed.")

    # Align
    aligned_img = cv2.warpPerspective(ref_img, h, (2480, 3508))
    
    return aligned_img


def findSquares(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold (convert black to white)
    _, binary = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lists to store detected squares
    large_squares = []
    small_squares = []

    # Process contours
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4: 
            x, y, w, h = cv2.boundingRect(approx)

            if 60 <= w <= 80 and 60 <= h <= 80:
                large_squares.append((x, y, w, h))
            elif 24 <= w <= 44 and 24 <= h <= 44:
                small_squares.append((x, y, w, h))

    return large_squares, small_squares


if __name__ == '__main__':
    # Read reference image
    ref_img_path = "../reference/reference.png" 
    ref_img = cv2.imread(ref_img_path)

    # Pipeline on scan images
    scanned_images_folder_path = "../data/scanned_images"
    aligned_images_folder_path = "../data/aligned_images"

    for image_file in os.listdir(scanned_images_folder_path):
        # Align image
        image_path = os.path.join(scanned_images_folder_path, image_file)
        image = cv2.imread(image_path)  
        image = cv2.resize(image, (2480, 3508))
        
        aligned_image = alignImages(ref_img, image)
        image_base_name = image_file.split(".")[0]
        aligned_image_path = aligned_images_folder_path + "/" + image_base_name + "_aligned.png"
        cv2.imwrite(aligned_image_path, aligned_image)

        # Split image
        large_squares, small_squares = findSquares(aligned_image)

        # Draw red boxes on large squares and label them
        for idx, (x, y, w, h) in enumerate(large_squares):
            cv2.rectangle(aligned_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(aligned_image, str(idx), (x + w + 10, y + h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw blue boxes on small squares and label them
        for idx, (x, y, w, h) in enumerate(small_squares):
            cv2.rectangle(aligned_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(aligned_image, str(idx), (x + w + 10, y + h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Save the result
        cv2.imwrite("detect1.png", aligned_image)


