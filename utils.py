from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares

def form_cam_matrix():
    """
    Form the camera matrix based on focal length, principal point, and image size.

    Returns
    -------
    The camera matrix. (np.ndarray)
    """
    focal_length = [615, 615]  
    principal_point = [320, 240]  
    image_size = (480, 640)  

    camera_matrix = np.array([[focal_length[0], 0, principal_point[0]],
                              [0, focal_length[1], principal_point[1]],
                              [0, 0, 1]], dtype=np.float32)
        
    return camera_matrix

K = form_cam_matrix()
T_x = 0.1  

# Define the translation vector
t = np.array([[T_x], [0], [0]])

# Construct initial projection matrices
P_l = np.hstack((K, np.zeros((3, 1))))
P_r = np.hstack((K, t))

def form_transf(R, t):
    """
    Makes a transformation matrix from the given rotation matrix and translation vector

    Parameters
    ----------
    R (ndarray): The rotation matrix
    t (list): The translation vector

    Returns
    -------
    T (ndarray): The transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


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


def temporal_features(img1, img2, features_1, max_error=4):
    lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    features_2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, features_1, None, **lk_params)
    trackable = np.squeeze(status.astype(bool))
    under_thresh = np.squeeze(np.where(err[trackable] < max_error, True, False))

    features_1 = features_1[trackable][under_thresh]
    features_2 = np.around(features_2[trackable][under_thresh])

    h, w, _ = img1.shape
    in_bounds = np.where(np.logical_and(features_2[:, 1] < h, features_2[:, 0] < w), True, False)
    features_1 = features_1[in_bounds]
    features_2 = features_2[in_bounds]

    return features_1, features_2


def reprojection_residuals(dof, feature_1, feature_2, Q1, Q2):
    """
    Calculate the residuals

    Parameters
    ----------
    dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)

    Returns
    -------
    residuals (ndarray): The residuals. In shape (2 * n_points * 2)
    """
    # Get the rotation vector
    r = dof[:3]
        # Create the rotation matrix from the rotation vector
    R, _ = cv2.Rodrigues(r)
    # Get the translation vector
    t = dof[3:]
    # Create the transformation matrix from the rotation matrix and translation vector
    transf = form_transf(R, t)

    # Create the projection matrix for the i-1'th image and i'th image
    f_projection = np.matmul(P_l, transf)
    b_projection = np.matmul(P_l, np.linalg.inv(transf))

    # Make the 3D points homogenize
    ones = np.ones((feature_1.shape[0], 1))
    Q1 = np.hstack([Q1, ones])
    Q2 = np.hstack([Q2, ones])

    # Project 3D points from i'th image to i-1'th image
    feature_1_pred = Q2.dot(f_projection.T)
    # Un-homogenize
    feature_1_pred = feature_1_pred[:, :2].T / feature_1_pred[:, 2]

    # Project 3D points from i-1'th image to i'th image
    feature_2_pred = Q1.dot(b_projection.T)
    # Un-homogenize
    feature_2_pred = feature_2_pred[:, :2].T / feature_2_pred[:, 2]

    # Calculate the residuals
    residuals = np.vstack([feature_1_pred - feature_1.T, feature_2_pred - feature_2.T]).flatten()
    return residuals


def estimate_pose(feature_1, feature_2, Q1, Q2, max_iter=100):
    early_termination_threshold = 5
    min_error = float('inf')
    early_termination = 0

    for _ in range(max_iter):
        sample_idx = np.random.choice(range(feature_1.shape[0]), 6)
        sample_q1, sample_q2, sample_Q1, sample_Q2 = feature_1[sample_idx], feature_2[sample_idx], Q1[sample_idx], Q2[sample_idx]

        in_guess = np.zeros(6)

        opt_res = least_squares(reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                args=(sample_q1, sample_q2, sample_Q1, sample_Q2))
        
        error = reprojection_residuals(opt_res.x, feature_1, feature_2, Q1, Q2)
        error = error.reshape((Q1.shape[0] * 2, 2))
        error = np.sum(np.linalg.norm(error, axis=1))

        if error < min_error:
            min_error = error
            out_pose = opt_res.x
            early_termination = 0
        else:
            early_termination += 1
        if early_termination == early_termination_threshold:
            # If we have not fund any better result in early_termination_threshold iterations
            break
    
    r = out_pose[:3]
    # Make the rotation matrix
    R, _ = cv2.Rodrigues(r)
    # Get the translation vector
    t = out_pose[3:]
    # Make the transformation matrix
    transformation_matrix = form_transf(R, t)

    return transformation_matrix