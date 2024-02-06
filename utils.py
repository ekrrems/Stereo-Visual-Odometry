from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import cv2

def pair_images(dir_path):
    directory = Path(dir_path)
    number_counts = {}

    for file in directory.iterdir():
        if file.is_file():
            number = int(file.stem.split('_')[1])
            number_counts[number] = number_counts.get(number, 0) + 1

    paired_numbers = [number for number, count in number_counts.items() if count > 1]

    left_images = [cv2.imread(str(file)) for file in directory.iterdir() if int(file.stem.split('_')[1]) in paired_numbers and file.name.startswith('L')]
    right_images = [cv2.imread(str(file)) for file in directory.iterdir() if int(file.stem.split('_')[1]) in paired_numbers and file.name.startswith('R')]

    return left_images, right_images


def find_features(left_img, right_img):
    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
    key_points_l, desc_l = sift.detectAndCompute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), None)
    key_points_r, desc_r = sift.detectAndCompute(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_l, desc_r, k=2)
    good_matches = []

    for m, n in matches:
        if m.distance < 0.60 * n.distance:
            good_matches.append(m)

    features_l = np.float32([key_points_l[m.queryIdx].pt for m in good_matches])
    features_r = np.float32([key_points_r[m.trainIdx].pt for m in good_matches])

    return key_points_l, key_points_r, features_l, features_r, good_matches


def plot_sequence(dir_path, plot_keypoints=False):
    left_images, right_images = pair_images(dir_path)

    for i in range(len(left_images)):
        if plot_keypoints:
            key_points_l, key_points_r, _, _, features =  find_features(left_images[i],
                                                                        right_images[i])
            keypoint_matches = cv2.drawMatches(left_images[i],
                                                key_points_l,
                                                right_images[i],
                                                key_points_r,
                                                features,
                                                None,
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
            cv2.imshow("Keypoint Matches", keypoint_matches)
            
        else:
            concat_image = cv2.hconcat([left_images[i], right_images[i]])
            cv2.imshow('concatenated Image', concat_image)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break