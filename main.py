import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.optimize import least_squares

# from lib.visualization import plotting
# from lib.visualization.video import play_trip

# read the images as a sequence
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


# Plot them as a pair
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

# plot_sequence('lamps', True)
        
images_l, images_r = pair_images('lamps')
print(images_l[0].shape, images_r[0].shape)

# Create disparity 

# block = 11
# P1 = block * block * 8
# P2 = block * block * 32
# disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
# disparities = [np.divide(disparity.compute(cv2.cvtColor(images_l[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(images_r[0], cv2.COLOR_BGR2GRAY)).astype(np.float32), 16)]
# print(disparities[0].shape)
# plt.imshow(disparities[0])
# plt.colorbar()
# plt.title('Disparity Map')
# plt.show()

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

    plt.show()

plot_disparities(images_l, images_r)

# disp = show_disparity(images_l[0], images_r[0], 35)
# plt.imshow(disp, cmap='gray')
# plt.axis('off')
# plt.show()

def form_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
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
    f_projection = np.matmul(self.P_l, transf)
    b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

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
'''









cv2.destroyAllWindows()