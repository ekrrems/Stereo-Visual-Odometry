'''This file contains helper functions'''
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import layout
from bokeh.models import Div

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
    """
    Pair images from the specified directory path.

    Parameters:
    - dir_path (str): Path to the directory containing the images.

    Returns:
    - left_images (list): List of left images.
    - right_images (list): List of right images.
    """
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
    """
    Find features in the left and right images using the SIFT algorithm.

    Parameters:
    - left_img (numpy.ndarray): Left image.
    - right_img (numpy.ndarray): Right image.

    Returns:
    - key_points_l (list): Keypoints in the left image.
    - key_points_r (list): Keypoints in the right image.
    - features_l (numpy.ndarray): Features in the left image.
    - features_r (numpy.ndarray): Features in the right image.
    - good_matches (list): List of good matches.
    """
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
    """
    Plot the image sequence from the specified directory path.

    Parameters:
    - dir_path (str): Path to the directory containing the image sequence.
    - plot_keypoints (bool): Flag to indicate whether to plot keypoints.

    Returns:
    None
    """
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


def temporal_features(img1_l, img2_l, features1_l, img1_r, img2_r, features1_r, max_error=4):
    """
    Track features from frame t to t+1 in both left and right images.

    Parameters:
    - img1_l (numpy.ndarray): Image frame at time t in the left camera.
    - img2_l (numpy.ndarray): Image frame at time t+1 in the left camera.
    - features1_l (numpy.ndarray): Features in the left image at time t.
    - img1_r (numpy.ndarray): Image frame at time t in the right camera.
    - img2_r (numpy.ndarray): Image frame at time t+1 in the right camera.
    - features1_r (numpy.ndarray): Features in the right image at time t.
    - max_error (int): Maximum error threshold for feature tracking.

    Returns:
    - features1_l (numpy.ndarray): Tracked features in the left image at time t.
    - features2_l (numpy.ndarray): Tracked features in the left image at time t+1.
    - features1_r (numpy.ndarray): Tracked features in the right image at time t.
    - features2_r (numpy.ndarray): Tracked features in the right image at time t+1.
    """
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    features2_l, status_l, err_l = cv2.calcOpticalFlowPyrLK(img1_l, img2_l, features1_l, None, **lk_params)
    trackable_l = np.squeeze(status_l.astype(bool))
    under_thresh_l = np.squeeze(np.where(err_l[trackable_l] < max_error, True, False))

    features1_l = features1_l[trackable_l][under_thresh_l]
    features2_l = np.around(features2_l[trackable_l][under_thresh_l])

    features2_r, status_r, err_r = cv2.calcOpticalFlowPyrLK(img1_r, img2_r, features1_r, None, **lk_params)
    trackable_r = np.squeeze(status_r.astype(bool))
    under_thresh_r = np.squeeze(np.where(err_r[trackable_r] < max_error, True, False))

    features1_r = features1_r[trackable_r][under_thresh_r]
    features2_r = np.around(features2_r[trackable_r][under_thresh_r])

    common_indices = np.intersect1d(np.arange(len(features2_l)), np.arange(len(features2_r)))

    features1_l = features1_l[common_indices]
    features2_l = features2_l[common_indices]

    features1_r = features1_r[common_indices]
    features2_r = features2_r[common_indices]

    return features1_l, features2_l, features1_r, features2_r


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
    r = dof[:3]
    R, _ = cv2.Rodrigues(r)
    t = dof[3:]
    transf = form_transf(R, t)

    f_projection = np.matmul(P_l, transf)
    b_projection = np.matmul(P_l, np.linalg.inv(transf))

    ones = np.ones((feature_1.shape[0], 1))
    Q1 = np.hstack([Q1, ones])
    Q2 = np.hstack([Q2, ones])

    feature_1_pred = Q2.dot(f_projection.T)
    feature_1_pred = feature_1_pred[:, :2].T / feature_1_pred[:, 2]

    feature_2_pred = Q1.dot(b_projection.T)
    feature_2_pred = feature_2_pred[:, :2].T / feature_2_pred[:, 2]

    residuals = np.vstack([feature_1_pred - feature_1.T, feature_2_pred - feature_2.T]).flatten()
    return residuals


def estimate_pose(feature_1, feature_2, Q1, Q2, max_iter=100):
    """
    Estimate the transformation matrix for the given features and disparity.

    Parameters:
    - feature_1 (numpy.ndarray): Features in the initial frame.
    - feature_2 (numpy.ndarray): Features in the subsequent frame.
    - Q1 (numpy.ndarray): Disparity map for the initial frame.
    - Q2 (numpy.ndarray): Disparity map for the subsequent frame.
    - max_iter (int): Maximum number of iterations for optimization.

    Returns:
    - transformation_matrix (numpy.ndarray): Transformation matrix.
    """
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
            break
    
    r = out_pose[:3]
    R, _ = cv2.Rodrigues(r)
    t = out_pose[3:]
    transformation_matrix = form_transf(R, t)

    return transformation_matrix


def visualize_paths(pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
    """
    Visualize the predicted path in 2D.

    Parameters
    ----------
    pred_path : np.ndarray
        Array containing the predicted path.
    html_tile : str, optional
        Title for the HTML output, by default "".
    title : str, optional
        Title for the visualization, by default "VO exercises".
    file_out : str, optional
        Output file name for the plot, by default "plot.html".
    """

    output_file(file_out, title=html_tile)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    pred_x, pred_z = pred_path.T
    source = ColumnDataSource(data=dict(px=pred_path[:, 0], pz=pred_path[:, 1]))

    fig = figure(title="Predicted Path (2D)", tools=tools, match_aspect=True, width_policy="max", toolbar_location="above",
                x_axis_label="x", y_axis_label="z", height=500, width=800,
                output_backend="webgl")

    fig.line("px", "pz", source=source, line_width=2, line_color="green", legend_label="Pred")
    fig.circle("px", "pz", source=source, size=8, color="green", legend_label="Pred")

    show(layout([Div(text=f"<h1>{title}</h1>"),
                [fig],
                ], sizing_mode='scale_width'))