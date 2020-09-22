import cv2
import numpy as np

import torch

from lib.models.vibe import VIBE_Demo
from lib.data_utils.kp_utils import get_spin_skeleton
from lib.data_utils import kp_utils
from lib.utils.demo_utils import download_ckpt
from lib.utils.vis import get_colors


colors_3d = {0: '#ff0000', 1: '#008000', 2: '#0000ff'}  # 0: head, 1: top, 2: bottom


names = [  #  get_spin_joint_names()
    'Nose',        # 0
    'Neck',        # 1
    'Shoulder_r',  # 2
    'Elbow_r',     # 3
    'RWrist_r',    # 4
    'Shoulder_l',  # 5
    'Elbow_l',     # 6
    'Wrist_l',     # 7
    'MidHip',      # 8
    'Hip_r',       # 9
    'Knee_r',      # 10
    'Ankle_r',     # 11
    'Hip_l',       # 12
    'Knee_l',      # 13
    'Ankle_l',     # 14
    'Eye_r',       # 15
    'Eye_l',       # 16
    'REar',        # 17
    'LEar',        # 18
    'LBigToe',     # 19
    'LSmallToe',   # 20
    'LHeel',       # 21
    'RBigToe',     # 22
    'RSmallToe',   # 23
    'RHeel',       # 24
    'rankle',         # 25
    'rknee',          # 26
    'rhip',           # 27
    'lhip',           # 28
    'lknee',          # 29
    'lankle',         # 30
    'rwrist',         # 31
    'relbow',         # 32
    'rshoulder',      # 33
    'lshoulder',      # 34
    'lelbow',         # 35
    'lwrist',         # 36
    'neck',           # 37
    'headtop',        # 38
    'hip',            # 39 'Pelvis (MPII)', # 39
    'thorax',         # 40 'Thorax (MPII)', # 40
    'Spine (H36M)',   # 41
    'Jaw (H36M)',     # 42
    'Head (H36M)',    # 43
    'nose',           # 44
    'leye',           # 45 'Left Eye', # 45
    'reye',           # 46 'Right Eye', # 46
    'lear',           # 47 'Left Ear', # 47
    'rear',           # 48 'Right Ear', # 48
]


def show_3d_pose(positions_3d, ax, radius=40):
    vals = positions_3d

    connections = get_spin_skeleton()

    LR = np.array([0, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0], dtype=int)

    for ind, (i,j) in enumerate(connections):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=colors_3d[LR[ind]])

    RADIUS = radius  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def show_2d_pose(positions_2d, image, dataset='spin', unnormalize=False, thickness=2):
    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()
    lcolor = get_colors()['blue'].tolist()

    skeleton = eval(f'kp_utils.get_{dataset}_skeleton')()
    common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
    for idx,pt in enumerate(positions_2d):
        cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)
        # cv2.putText(image, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

    for i,(j1,j2) in enumerate(skeleton):
        if dataset == 'common':
            color = rcolor if common_lr[i] == 0 else lcolor
        else:
            color = lcolor if i % 2 == 0 else rcolor
        pt1, pt2 = (positions_2d[j1, 0], positions_2d[j1, 1]), (positions_2d[j2, 0], positions_2d[j2, 1])
        cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    return image


def get_demo_vibe_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    return model
