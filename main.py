import numpy as np
import cv2
from tqdm import tqdm
import os
from pathlib import Path
import utils

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import layout
from bokeh.models import Div

## Stereo Feature Matching
# utils.plot_sequence('lamps', True)
imgs_l, imgs_r = utils.pair_images('lamps')
# img0_l, img0_r = imgs_l[0], imgs_r[0]
# img1_l, img1_r = imgs_l[1], imgs_r[1]

# ## Stereo Feature Matching
# key_points_l, key_points_r, features_l, features_r, good_matches = utils.find_features(img0_l, img0_r)

# # plot the stereo match 
# matches_img = cv2.drawMatches(img0_l,
#                               key_points_l,
#                               img0_r,
#                               key_points_r,
#                               good_matches,
#                               None,
#                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# # cv2.imshow('matches_img', matches_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# print(' okay this is working')

# ## Temporal Feature Matching


# features0_l, features1_l = utils.temporal_features(img0_l, img1_l, features_l)
# print(features0_l.shape, features1_l.shape)

# features0_r, features1_r = utils.temporal_features(img0_r, img1_r, features_r)
# # Calculate 3D points

# print(features0_l.shape, features0_r.shape, features1_l.shape, features1_r.shape)

K = utils.form_cam_matrix()
T_x = 0.1  # Example baseline distance in meters

# Define the translation vector
t = np.array([[T_x], [0], [0]])

# Construct initial projection matrices
P_l = np.hstack((K, np.zeros((3, 1))))
P_r = np.hstack((K, t))


def calculate_3d(feat1_l, feat1_r, feat2_l, feat2_r):
    Q1 = cv2.triangulatePoints(P_l, P_r, feat1_l.T, feat1_r.T)
    Q1 = np.transpose(Q1[:3] / Q1[3])

    Q2 = cv2.triangulatePoints(P_l, P_r, feat2_l.T, feat2_r.T)
    Q2 = np.transpose(Q2[:3] / Q2[3])

    return Q1, Q2


# Q1, Q2 = calculate_3d(features0_l, features0_r, features1_l, features1_r)

# transformation_matrix = utils.estimate_pose(features0_l, features1_l, Q1, Q2)
# print(transformation_matrix)

# imgs_l, imgs_r = utils.pair_images('lamps')
# i = 1

# img1_l, img2_l = imgs_l[i - 1:i + 1]
# img1_r, img2_r = imgs_r[i - 1:i + 1]

# key_points_l, key_points_r, features_l, features_r, good_matches = utils.find_features(img1_l, img1_r)

# features1_l, features2_l = utils.temporal_features(img1_l, img2_l, features_l)

# features1_r, features2_r = utils.temporal_features(img1_r, img2_r, features_r)


# Q1, Q2 = calculate_3d(features1_l, features1_r, features2_l, features2_r)
# print(Q1.shape, Q2.shape)

# transformation_matrix = utils.estimate_pose(features1_l, features2_l, Q1, Q2)
# print(transformation_matrix)\

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

def get_camera_pose(i):
    img1_l, img2_l = imgs_l[i - 1:i + 1]
    img1_r, img2_r = imgs_r[i - 1:i + 1]

    _, _, features_l, features_r, _ = utils.find_features(img1_l, img1_r)

    features1_l, features2_l = utils.temporal_features(img1_l, img2_l, features_l)

    features1_r, features2_r = utils.temporal_features(img1_r, img2_r, features_r)

    Q1, Q2 = calculate_3d(features1_l, features1_r, features2_l, features2_r)

    transformation_matrix = utils.estimate_pose(features1_l, features2_l, Q1, Q2)

    return transformation_matrix

# transformation_matrix = get_camera_pose(1)

# def main
data_dir = 'lamps'
imgs_l, imgs_r = utils.pair_images(data_dir)

estimated_path = []
for i in range(len(imgs_l)):
    if i < 1:
        cur_pose = np.eye(4)
    else:
        print(i)
        
        transformation_matrix = get_camera_pose(i)
        cur_pose = np.matmul(cur_pose, transformation_matrix)
        # except (cv2.error, ValueError) as e:
            # continue
    estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
visualize_paths(estimated_path, "Visual Odometry")
# Tasks fo 20/01/2024
'''
*** Troubleshooting
Check out if something gives and error solve it and then continue

1. Create Odometry path: https://www.youtube.com/watch?v=WV3ZiPqd2G4
2. Density Map with the 3d point clouds 


# docker commands
docker build -t stereo_depth .
docker images


Follow this pdf
https://www.cs.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A12-StereoVO.pdf
https://www-robotics.jpl.nasa.gov/media/documents/howard_iros08_visodom.pdf
https://github.com/niconielsen32/ComputerVision/blob/master/VisualOdometry/stereo_visual_odometry.py

'''









cv2.destroyAllWindows()