import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

def form_cam_matrix():
    """
    Form the camera matrix based on focal length, principal point.

    Returns
    -------
    np.ndarray
    The camera matrix.
    """
    focal_length = [615, 615]  
    principal_point = [320, 240]  

    camera_matrix = np.array([[focal_length[0], 0, principal_point[0]],
                                [0, focal_length[1], principal_point[1]],
                                [0, 0, 1]], dtype=np.float32)
        
    return camera_matrix


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


def plot_sequence(dir_path, plot_features=False):
    left_images, right_images = pair_images(dir_path)

    for i in range(len(left_images)):
        if plot_features:
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


def compute_disparity(image_l, image_r, block_size=5) :
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=block_size)

    disparity = stereo.compute(cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY))

    min = disparity.min()
    max = disparity.max()

    disparity = np.uint8(255 * (disparity - min) / (max - min))

    return disparity


def plot_disparities(images_l, images_r):
    disparity_maps = []

    for img_l, img_r in tqdm(zip(images_l, images_r), total=len(images_l), desc="Creating Disparity Maps"):
        disparity_map = compute_disparity(img_l, img_r, 25)
        disparity_maps.append(disparity_map)

    # Plot the disparity maps
    for i, disparity_map in enumerate(disparity_maps):
        plt.imshow(disparity_map, cmap='gray')
        plt.axis('off')
        plt.title(f'Disparity Map {i}')

        plt.show(block=False)
        plt.pause(0.3)  
        plt.clf()

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    plt.show()


def form_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

K = form_cam_matrix()
P = np.matmul(K, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

def reprojection_residuals(dof, q1, q2, Q1, Q2):
    """
    Calculate the residuals

    Parameters
    ----------
    dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
    q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
    q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
    Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
    Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

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
    transf = form_transform(R, t)

    # Create the projection matrix for the i-1'th image and i'th image
    f_projection = np.matmul(P, transf)
    b_projection = np.matmul(P, np.linalg.inv(transf))

    # Make the 3D points homogenize
    ones = np.ones((q1.shape[0], 1))
    Q1 = np.hstack([Q1, ones])
    Q2 = np.hstack([Q2, ones])

    # Project 3D points from i'th image to i-1'th image
    q1_pred = Q2.dot(f_projection.T)
    # Un-homogenize
    q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

    # Project 3D points from i-1'th image to i'th image
    q2_pred = Q1.dot(b_projection.T)
    # Un-homogenize
    q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

    # Calculate the residuals
    residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
    return residuals


def track_keypoints(self, img1, img2, kp1, max_error=4):
    """
    Tracks the keypoints between frames

    Parameters
    ----------
    img1 (ndarray): i-1'th image. Shape (height, width)
    img2 (ndarray): i'th image. Shape (height, width)
    kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
    max_error (float): The maximum acceptable error

    Returns
    -------
    trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
    trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
    """
    # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
    trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

    # Use optical flow to find tracked counterparts
    trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

    # Convert the status vector to boolean so we can use it as a mask
    trackable = st.astype(bool)

    # Create a maks there selects the keypoints there was trackable and under the max error
    under_thresh = np.where(err[trackable] < max_error, True, False)

    # Use the mask to select the keypoints
    trackpoints1 = trackpoints1[trackable][under_thresh]
    trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

    # Remove the keypoints there is outside the image
    h, w = img1.shape
    in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
    trackpoints1 = trackpoints1[in_bounds]
    trackpoints2 = trackpoints2[in_bounds]

    return trackpoints1, trackpoints2


def calculate_right_qs(q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
    """
    Calculates the right keypoints (feature points)

    Parameters
    ----------
    q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
    q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
    disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
    disp2 (ndarray): Disparity i'th image per. Shape (height, width)
    min_disp (float): The minimum disparity
    max_disp (float): The maximum disparity

    Returns
    -------
    q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
    q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
    q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
    q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
    """
    def get_idxs(q, disp):
        q_idx = q.astype(int)
        disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
        return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
    # Get the disparity's for the feature points and mask for min_disp & max_disp
    disp1, mask1 = get_idxs(q1, disp1)
    disp2, mask2 = get_idxs(q2, disp2)
        
    # Combine the masks 
    in_bounds = np.logical_and(mask1, mask2)
        
    # Get the feature points and disparity's there was in bounds
    q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
    # Calculate the right feature points 
    q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
    q1_r[:, 0] -= disp1
    q2_r[:, 0] -= disp2
        
    return q1_l, q1_r, q2_l, q2_r


def calc_3d(q1_l, q1_r, q2_l, q2_r):
    Q1 = cv2.triangulatePoints(P_l, P_r, q1_l.T, q1_r.T)
    Q1 = np.transpose(Q1[:3] / Q1[3])

    Q2 = cv2.triangulatePoints(P_l, P_r, q2_l.T, q2_r.T)
    Q2 = np.transpose(Q2[:3] / Q2[3])

    return Q1, Q2


def estimate_pose(q1, q2, Q1, Q2, max_iter=100):
    """
    Estimates the transformation matrix

    Parameters
    ----------
    q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
    q2 (ndarray): Feature points in i'th image. Shape (n, 2)
    Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
    Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
    max_iter (int): The maximum number of iterations

    Returns
    -------
    transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
    """
    early_termination_threshold = 5

    # Initialize the min_error and early_termination counter
    min_error = float('inf')
    early_termination = 0

    for _ in range(max_iter):
        # Choose 6 random feature points
        sample_idx = np.random.choice(range(q1.shape[0]), 6)
        sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

        # Make the start guess
        in_guess = np.zeros(6)
        # Perform least squares optimization
        opt_res = least_squares(reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

        # Calculate the error for the optimized transformation
        error = reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
        error = error.reshape((Q1.shape[0] * 2, 2))
        error = np.sum(np.linalg.norm(error, axis=1))

        # Check if the error is less the the current min error. Save the result if it is
        if error < min_error:
            min_error = error
            out_pose = opt_res.x
            early_termination = 0
        else:
            early_termination += 1
        if early_termination == early_termination_threshold:
            # If we have not fund any better result in early_termination_threshold iterations
            break

    # Get the rotation vector
    r = out_pose[:3]
    # Make the rotation matrix
    R, _ = cv2.Rodrigues(r)
    # Get the translation vector
    t = out_pose[3:]
    # Make the transformation matrix
    transformation_matrix = form_transform(R, t)
    return transformation_matrix


def get_pose(i):
    """
    Calculates the transformation matrix for the i'th frame

    Parameters
    ----------
    i (int): Frame index

    Returns
    -------
    transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
    """
    # Get the i-1'th image and i'th image
    img1_l, img2_l = images_l[i - 1:i + 1]

    # Get teh tiled keypoints
    kp1_l = get_tiled_keypoints(img1_l, 10, 20)

    # Track the keypoints
    tp1_l, tp2_l = track_keypoints(img1_l, img2_l, kp1_l)

    # Calculate the disparitie
    disparities.append(np.divide(disparity.compute(img2_l, images_r[i]).astype(np.float32), 16))

    # Calculate the right keypoints
    tp1_l, tp1_r, tp2_l, tp2_r = calculate_right_qs(tp1_l, tp2_l, disparities[i - 1], disparities[i])

    # Calculate the 3D points
    Q1, Q2 = calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

    # Estimate the transformation matrix
    transformation_matrix = estimate_pose(tp1_l, tp2_l, Q1, Q2)
    return transformation_matrix