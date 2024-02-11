import numpy as np
import cv2
import utils

K = utils.form_cam_matrix()
T_x = 0.1  

t = np.array([[T_x], [0], [0]])

P_l = np.hstack((K, np.zeros((3, 1))))
P_r = np.hstack((K, t))
data_dir = 'lamps'
imgs_l, imgs_r = utils.pair_images(data_dir)

def calculate_3d(feat1_l, feat1_r, feat2_l, feat2_r):
    """
    Calculate 3D points using triangulation for corresponding features in stereo images.

    Parameters:
    - feat1_l (numpy.ndarray): Features in the left image at time t.
    - feat1_r (numpy.ndarray): Features in the right image at time t.
    - feat2_l (numpy.ndarray): Features in the left image at time t+1.
    - feat2_r (numpy.ndarray): Features in the right image at time t+1.
    - P_l (numpy.ndarray): Projection matrix for the left camera.
    - P_r (numpy.ndarray): Projection matrix for the right camera.

    Returns:
    - Q1 (numpy.ndarray): 3D points corresponding to feat1_l and feat1_r.
    - Q2 (numpy.ndarray): 3D points corresponding to feat2_l and feat2_r.
    """
    Q1 = cv2.triangulatePoints(P_l, P_r, feat1_l.T, feat1_r.T)
    Q1 = np.transpose(Q1[:3] / Q1[3])

    Q2 = cv2.triangulatePoints(P_l, P_r, feat2_l.T, feat2_r.T)
    Q2 = np.transpose(Q2[:3] / Q2[3])

    return Q1, Q2


def get_camera_pose(i):
    """
    Estimate the camera pose for the i'th frame based on stereo image pairs.

    Parameters:
    - i (int): Frame index.

    Returns:
    - transformation_matrix (numpy.ndarray or None): The transformation matrix if successful, None otherwise.
    """
    img1_l, img2_l = imgs_l[i - 1:i + 1]
    img1_r, img2_r = imgs_r[i - 1:i + 1]

    _, _, features_l, features_r, _ = utils.find_features(img1_l, img1_r)

    features1_l, features2_l, features1_r, features2_r = utils.temporal_features(img1_l, img2_l, features_l, img1_r, img2_r, features_r)

    if features1_l.shape[0] < 3:
        return None

    Q1, Q2 = calculate_3d(features1_l, features1_r, features2_l, features2_r)

    transformation_matrix = utils.estimate_pose(features1_l, features2_l, Q1, Q2) 

    return transformation_matrix

def main():
    """
    Main function to estimate the camera path using visual odometry.
    """
    # utils.plot_sequence('lamps', True) # Uncomment to see the features
    estimated_path = []
    for i in range(len(imgs_l)):
        if i < 1:
            cur_pose = np.eye(4)
        else:
            print(i)
            try:
                transformation_matrix = get_camera_pose(i)
                if transformation_matrix is None:
                    continue
                cur_pose = np.matmul(cur_pose, transformation_matrix)
            except:
                continue
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    utils.visualize_paths(estimated_path, "Visual Odometry") 

if __name__ =='__main__':
    main()
