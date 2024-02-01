import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import layout
from bokeh.models import Div
import utils

# from lib.visualization import plotting
# from lib.visualization.video import play_trip

K = utils.form_cam_matrix()
# P = np.matmul(K, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
lk_params = dict(winSize=(15, 15),
                 flags=cv2.MOTION_AFFINE,
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

data_dir = 'lamps'

images_l, images_r = utils.pair_images(data_dir)

# img1, img2 = images_l[0:2]
# imgr_1, imgr_2 = images_r[0:2]

# print(img1.shape)
# _, _, kp1_l, _, _ = utils.find_features(img1, img2)

# disp1 = utils.compute_disparity(img1, imgr_1)
# disp2 = utils.compute_disparity(img2, imgr_2)
# print(disp1.shape)


# Apply track keypoint function to my code
# trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kpl_1), axis=1)
def track_keypoints(img1, img2, kp1, max_error=4):
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
    kp2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, kp1, None, **lk_params)
    trackable = st.astype(bool)
    trackable = trackable.squeeze(axis=1)

    under_thresh = np.where(err[trackable] < max_error, True, False)
    under_thresh = under_thresh.squeeze(axis=1)

    kp1 = kp1[trackable][under_thresh]
    kp2 = kp2[trackable][under_thresh]

    h, w, _ = img1.shape
    in_bounds = np.where(np.logical_and(kp2[:, 1] < h, kp2[:, 0] < w), True, False)
    kp1 = kp1[in_bounds]
    kp2 = kp2[in_bounds]

    return kp1, kp2



def plot_tracked_keypoints(img1, img2, trackpoints1, trackpoints2):
    # Create a new image by concatenating img1 and img2 horizontally
    concatenated_img = cv2.hconcat([img1, img2])

    # Draw tracked keypoints on the concatenated image
    for point1, point2 in zip(trackpoints1, trackpoints2):
        # Convert points to integers for drawing
        point1 = tuple(map(int, point1))
        point2 = tuple(map(int, point2))
        # Offset the x-coordinate of the second set of keypoints
        point2 = (point2[0] + img1.shape[1], point2[1])
        # Draw circles around the keypoints
        cv2.circle(concatenated_img, point1, 5, (0, 255, 0), -1)  # Green for keypoints in img1
        cv2.circle(concatenated_img, point2, 5, (0, 0, 255), -1)  # Red for keypoints in img2

    # Display the concatenated image with tracked keypoints
    cv2.imshow('Tracked Keypoints', concatenated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# calculate right image keypoints
def calculate_right_keypoints(kp1, kp2, disp1, disp2, min_disp=0, max_disp=100):
    def get_idxs(q, disp):
        q_idx = q.astype(int)
        disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
        return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

    disp1, mask1 = get_idxs(kp1, disp1)
    disp2, mask2 = get_idxs(kp2, disp2)
        
        # Combine the masks 
    in_bounds = np.logical_and(mask1, mask2)
            
            # Get the feature points and disparity's there was in bounds
    kp1_l, kp2_l, disp1, disp2 = kp1[in_bounds], kp2[in_bounds], disp1[in_bounds], disp2[in_bounds]
            
            # Calculate the right feature points 
    kp1_r, kp2_r = np.copy(kp1_l), np.copy(kp2_l)
    kp1_r[:, 0] -= disp1
    kp2_r[:, 0] -= disp2

    return kp1_l, kp1_r, kp2_l, kp2_r


# kp1_l,kp2_l = track_keypoints(img1, img2, kp1_l)
# kp1_l, kp1_r, kp2_l, kp2_r = calculate_right_keypoints(kp1_l,kp2_l, disp1, disp2)

# Calculate the 3d points 
# def calc_3d
# Q1 = cv2.triangulatePoints(P, P, q1_l.T, q1_r.T)
# Q1 = np.transpose(Q1[:3] / Q1[3])


# Q2 = cv2.triangulatePoints(P, P, q2_l.T, q2_r.T)
# Q2 = np.transpose(Q2[:3] / Q2[3])

# print(Q1.shape, Q2.shape)


##### Estimate Pose

'''
So we have first and second iamge feature poitns or keypoints lets ge the pose value using those values 
'''

# Left transform matrix

def decomp_essential_mat(E, q1_left, q2_left, q1_right, q2_right):
    global K

    R1, R2, t = cv2.decomposeEssentialMat(E)
    T1 = utils.form_transform(R1, np.ndarray.flatten(t))
    T2 = utils.form_transform(R2, np.ndarray.flatten(t))
    T3 = utils.form_transform(R1, np.ndarray.flatten(-t))
    T4 = utils.form_transform(R2, np.ndarray.flatten(-t))
    transformations = [T1, T2, T3, T4]

    K = np.concatenate((K, np.zeros((3, 1))), axis=1)

    projections = [K @ T1, K @ T2, K @ T3, K @ T4]

    positives = []
    for P, T in zip(projections, transformations):
        hom_Q1_left = cv2.triangulatePoints(P, P, q1_left.T, q2_left.T)
        hom_Q2_left = T @ hom_Q1_left
        hom_Q1_right = cv2.triangulatePoints(P, P, q1_right.T, q2_right.T)
        hom_Q2_right = T @ hom_Q1_right

        Q1_left = hom_Q1_left[:3, :] / hom_Q1_left[3, :]
        Q2_left = hom_Q2_left[:3, :] / hom_Q2_left[3, :]
        Q1_right = hom_Q1_right[:3, :] / hom_Q1_right[3, :]
        Q2_right = hom_Q2_right[:3, :] / hom_Q2_right[3, :]

        total_sum = sum((Q2_left[2, :] > 0) & (Q1_left[2, :] > 0) &
                        (Q2_right[2, :] > 0) & (Q1_right[2, :] > 0))
        relative_scale = np.mean(
            np.linalg.norm(Q1_left.T[:-1] - Q1_left.T[1:], axis=-1) /
            np.linalg.norm(Q2_left.T[:-1] - Q2_left.T[1:], axis=-1) +
            np.linalg.norm(Q1_right.T[:-1] - Q1_right.T[1:], axis=-1) /
            np.linalg.norm(Q2_right.T[:-1] - Q2_right.T[1:], axis=-1))

        positives.append(total_sum + relative_scale)

    max_idx = np.argmax(positives)
    if max_idx == 2:
        return utils.form_transform(R1, np.ndarray.flatten(-t))
    elif max_idx == 3:
        return utils.form_transform(R2, np.ndarray.flatten(-t))
    elif max_idx == 0:
        return utils.form_transform(R1, np.ndarray.flatten(t))
    elif max_idx == 1:
        return utils.form_transform(R2, np.ndarray.flatten(t))
    

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


# Essential, mask = cv2.findEssentialMat(kp1_l, kp2_l, K)
# transform_matrix = decomp_essential_mat(Essential, kp1_l, kp2_l, kp1_r, kp2_r)

# print(transform_matrix)

def get_transf(i):
    K = utils.form_cam_matrix()
    img1_l, img2_l = images_l[i - 1:i + 1]
    img1_r, img2_r = images_r[i - 1:i + 1]

    _, _, kp1_l, _, _ = utils.find_features(img1_l, img2_l)
    kp1_l, kp2_l = track_keypoints(img1_l, img2_l, kp1_l)

    disp1 = utils.compute_disparity(img1_l, img1_r)
    disp2 = utils.compute_disparity(img2_l, img2_r)

    kp1_l, kp1_r, kp2_l, kp2_r = calculate_right_keypoints(kp1_l, kp2_l, disp1, disp2)

    Essential, _ = cv2.findEssentialMat(kp1_l, kp2_l, K)
    transform_matrix = decomp_essential_mat(Essential, kp1_l, kp2_l, kp1_r, kp2_r)
    
    return transform_matrix
# print(get_transf(1))

def main():
    
    estimated_path = []
    data_dir = 'lamps'
    images_l, images_r = utils.pair_images(data_dir)

    for i, img in enumerate(tqdm(images_l, unit="pose")):
        if i < 1:
            cur_pose = np.eye(4)
        else:
            print(i)
            try:
                transf = get_transf(i)
                cur_pose = np.matmul(cur_pose, transf)
            except (cv2.error, ValueError) as e:
                continue
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    visualize_paths(estimated_path, "Visual Odometry")

# print('main is running')
main()

# def main():
# utils.plot_sequence(data_dir, True) #uncomment not to see the plot




































# early_termination_threshold = 5
# max_iter = 100

#         # Initialize the min_error and early_termination counter
# min_error = float('inf')
# early_termination = 0
# for _ in range(max_iter):
#             # Choose 6 random feature points
#     sample_idx = np.random.choice(range(q1_l.shape[0]), 6)
#     sample_q1, sample_q2, sample_Q1, sample_Q2 = q1_l[sample_idx], q2_l[sample_idx], Q1[sample_idx], Q2[sample_idx]

#             # Make the start guess
#     in_guess = np.ones(6)
#             # Perform least squares optimization
#     opt_res = least_squares(utils.reprojection_residuals, in_guess, method='lm', max_nfev=200,
#                             args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

#             # Calculate the error for the optimized transformation
#     error = utils.reprojection_residuals(opt_res.x, trackpoints1, trackpoints2, Q1, Q2)
#     error = error.reshape((Q1.shape[0] * 2, 2))
#     error = np.sum(np.linalg.norm(error, axis=1))

#             # Check if the error is less the the current min error. Save the result if it is
#     if error < min_error:
#         min_error = error
#         out_pose = opt_res.x
#         early_termination = 0
#     else:
#         early_termination += 1
#     if early_termination == early_termination_threshold:
#         # If we have not fund any better result in early_termination_threshold iterations
#         break

# r = out_pose[:3]
#         # Make the rotation matrix
# R, _ = cv2.Rodrigues(r)
#         # Get the translation vector
# t = out_pose[3:]
#         # Make the transformation matrix
# transformation_matrix = utils.form_transform(R, t)
# print(transformation_matrix)

# utils.plot_sequence('lamps', True)
        
# images_l, images_r = utils.pair_images('lamps')
# print(images_l[0].shape, images_r[0].shape)


# plot_disparities(images_l, images_r)











# cv2.imshow('left', left[100])
# cv2.imshow('right', right[100])
# cv2.imshow('conaat', concatenated_image)

# cv2.waitKey(0)


# Tasks of 19/01/2024
# Extract features (done)
# Plot the extracted features as pair (done)
# Show the epipolr geometry (optional)

# Tasks fo 20/01/2024
'''
1. Create Odometry path: 
2. Density Map with the (Done)
3. 3d point clouds 

# docker commands
docker build -t stereo_depth .
docker images

links::
https://github.com/PacktPublishing/OpenCV-with-Python-By-Example/blob/master/Chapter11/stereo_match.py
https://www.youtube.com/watch?v=WV3ZiPqd2G4
https://github.com/niconielsen32/ComputerVision/blob/master/VisualOdometry/stereo_visual_odometry.py
https://github.com/sakshamjindal/Stereo-Visual-SLAM-Odometry
https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/

Study:
https://cmsc426.github.io/sfm/
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=96f3c17a45a4b947bf2549893035ef6e163c876b follow this way
'''









cv2.destroyAllWindows()