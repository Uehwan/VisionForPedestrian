import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from lib.models.spin import perspective_projection
from utils import show_2d_pose, show_3d_pose, get_demo_vibe_model, names


# joint 3d extracted from vibe
# for the demo purpose, we excerpt only one frame pose from the full pose set
joints_3d = np.array([[ 1.0376830e-02, -8.5307413e-01, -2.2456938e-01],
                      [ 3.4354139e-02, -7.4853122e-01, -5.6185231e-02],
                      [-1.3761541e-01, -6.4946252e-01,  5.6735519e-04],
                      [-1.7643900e-01, -4.1796446e-01,  3.7669029e-02],
                      [-2.0997077e-01, -1.8824263e-01, -2.6238233e-02],
                      [ 1.8785083e-01, -6.1472708e-01, -7.3772341e-02],
                      [ 2.1103142e-01, -3.7858719e-01, -5.0772727e-03],
                      [ 2.1731015e-01, -1.4587167e-01, -4.3980490e-02],
                      [-4.2087883e-03, -2.3857084e-01,  1.6089346e-02],
                      [-6.0551435e-02, -1.5773880e-01,  4.4132568e-02],
                      [-6.0389444e-02,  2.0222612e-01,  7.6554462e-02],
                      [ 7.3558621e-02,  4.6476442e-01,  2.9823783e-01],
                      [ 4.5974620e-02, -1.4664924e-01,  1.3096088e-02],
                      [-2.4388984e-02,  1.9610748e-01, -2.7028304e-02],
                      [-1.0893974e-01,  5.2832466e-01,  1.1038256e-01],
                      [-8.5491240e-03, -8.9836341e-01, -1.8675199e-01],
                      [ 5.4729097e-02, -8.9154041e-01, -2.0344855e-01],
                      [-2.3720842e-02, -8.9190328e-01, -8.6340480e-02],
                      [ 1.1353934e-01, -8.7709087e-01, -1.2293894e-01],
                      [-1.7851977e-01,  5.7542825e-01, -6.2786564e-02],
                      [-1.0924992e-01,  6.0570955e-01, -3.2820396e-02],
                      [-9.3096666e-02,  5.5889976e-01,  1.6470701e-01],
                      [ 3.2815441e-02,  5.8935469e-01,  1.5640225e-01],
                      [-1.3445705e-02,  5.8176887e-01,  2.2760403e-01],
                      [ 9.3937069e-02,  4.7325325e-01,  3.6080492e-01],
                      [ 7.3558621e-02,  4.6476442e-01,  2.9823783e-01],
                      [-6.0389444e-02,  2.0222612e-01,  7.6554462e-02],
                      [-1.1179424e-01, -2.7067995e-01,  4.0088896e-02],
                      [ 9.7676039e-02, -2.4804579e-01, -2.5573831e-02],
                      [-2.4388984e-02,  1.9610748e-01, -2.7028304e-02],
                      [-1.0893974e-01,  5.2832466e-01,  1.1038256e-01],
                      [-2.0997077e-01, -1.8824263e-01, -2.6238233e-02],
                      [-1.7643900e-01, -4.1796446e-01,  3.7669029e-02],
                      [-1.3761541e-01, -6.4946252e-01,  5.6735519e-04],
                      [ 1.8785083e-01, -6.1472708e-01, -7.3772341e-02],
                      [ 2.1103142e-01, -3.7858719e-01, -5.0772727e-03],
                      [ 2.1731015e-01, -1.4587167e-01, -4.3980490e-02],
                      [ 3.4767319e-02, -7.4252123e-01, -6.4374581e-02],
                      [ 5.1507324e-02, -1.0153735e+00, -1.3421991e-01],
                      [-1.0844022e-03, -2.6219279e-01,  2.3834165e-02],
                      [ 3.0295696e-02, -6.7019922e-01, -3.8598716e-02],
                      [ 1.9880773e-02, -5.1308668e-01,  4.8420206e-03],
                      [ 3.4119926e-02, -8.2836223e-01, -1.4991075e-01],
                      [ 4.8211794e-02, -9.5452988e-01, -1.1687765e-01],
                      [ 1.0376830e-02, -8.5307413e-01, -2.2456938e-01],
                      [ 5.4729097e-02, -8.9154041e-01, -2.0344855e-01],
                      [-8.5491240e-03, -8.9836341e-01, -1.8675199e-01],
                      [ 1.1353934e-01, -8.7709087e-01, -1.2293894e-01],
                      [-2.3720842e-02, -8.9190328e-01, -8.6340480e-02],
                      [ 1.5342210e-02, -5.0092012e-01, -1.9038547e-02],
                      [ 4.6426937e-02, -5.1588941e-01,  1.1005889e-01]], dtype=np.float32)


bbox = np.array([611.42108154,  70.38444901,  66.25693402,  66.25693402])
pred_cam_t = torch.tensor([[-0.0688,  0.0581, 42.4410]])
focal_length = 5000
camera_center = torch.zeros(2).unsqueeze(0)
rotation = torch.eye(3).unsqueeze(0)


if __name__ == "__main__":
    test_img = './data/test_images/pose_example.png'
    img = cv2.cvtColor(cv2.imread(test_img), cv2.COLOR_BRG2RGB)

    # extraction of body orientation
    joints_3d_shoulder_left = joints_3d[5]
    joints_3d_shoulder_right = joints_3d[2]
    joints_3d_hip_middle = joints_3d[8]

    joints_3d_body_center = (joints_3d_shoulder_left + joints_3d_shoulder_right + joints_3d_hip_middle) / 3
    
    body_orientation_starting_point = joints_3d_body_center
    joints_3d_body_direction = body_orientation_starting_point - np.cross(
        (joints_3d_shoulder_left - joints_3d_hip_middle),
        (joints_3d_shoulder_right - joints_3d_hip_middle)
    ).reshape(1, 3)
    joints_3d = np.vstack((joints_3d, body_orientation_starting_point.reshape(1, 3), joints_3d_body_direction))

    joints_3d_tensor = torch.from_numpy(joints_3d).unsqueeze(0)

    joints_2d = perspective_projection(
        joints_3d_tensor,
        rotation,
        pred_cam_t,
        focal_length,
        camera_center
    ).squeeze().numpy()

    # extraction of head orientation
    mid_front = (joints_2d[15:16] + joints_2d[16:17]) / 2  # (Leye + Reye) / 2
    mid_back = (joints_2d[42:43] + joints_2d[43:44]) / 2  # (Jaw + Head) / 2

    joints_2d = np.vstack((joints_2d, mid_front, mid_back))

    # scale back to the original image
    scaled_joints_2d = joints_2d * bbox[2:] / 224 + bbox[:2]
    scaled_joints_2d = scaled_joints_2d.astype(int)
    
    for i in range(len(scaled_joints_2d)):
        img = cv2.circle(img, (scaled_joints_2d[i, 0], scaled_joints_2d[i, 1]), 2, (0, 0, 255), -1)
    
    # drawing head orientation
    x_start, y_start = scaled_joints_2d[51]
    x_delta, y_delta = 16 * (scaled_joints_2d[51] - scaled_joints_2d[52])
    img = cv2.arrowedLine(img, (x_start, y_start), (x_start+x_delta, y_start+y_delta), (0, 0, 255), 4, tipLength=0.5)

    # drawing body orientation
    x_start, y_start = scaled_joints_2d[49]
    x_delta, y_delta = 16 * (scaled_joints_2d[49] - scaled_joints_2d[50])
    img = cv2.arrowedLine(img, (x_start, y_start), (x_start+x_delta, y_start+y_delta), (0, 255, 0), 6, tipLength=0.5)

    plt.close('all')
    show_2d_pose(scaled_joints_2d, img, dataset='spin', unnormalize=False)
    plt.imshow(img)
    plt.show()

    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    show_3d_pose(joints_3d, ax, 1)
    ax.view_init(-75, -90)

    component_right = [2, 3, 4, 10, 11, 15]
    component_left = [5, 6, 7, 13, 14, 16]

    for i in component_right:
        ax.scatter(joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2], marker='o', s=25, c="#FFC45D")
        ax.text(joints_3d[i, 0]-0.35, joints_3d[i, 1], joints_3d[i, 2], names[i], size=10)

    for i in component_left:
        ax.scatter(joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2], marker='o', s=25, c="#FFC45D")
        ax.text(joints_3d[i, 0]+0.1, joints_3d[i, 1], joints_3d[i, 2], names[i], size=10)

    for i in [9]:
        ax.scatter(joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2], marker='o', s=25, c="#FFC45D")
        ax.text(joints_3d[i, 0]-0.1, joints_3d[i, 1]+0.1, joints_3d[i, 2], names[i], size=10)

    for i in [12]:
        ax.scatter(joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2], marker='o', s=25, c="#FFC45D")
        ax.text(joints_3d[i, 0], joints_3d[i, 1]+0.1, joints_3d[i, 2], names[i], size=10)

    plt.show()
