import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

import torch

from lib.models.spin import perspective_projection


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    angle = 360 - angle if v1[1] < 0 else angle
    return angle


def most_frequent(list_of_elements):
    if len(list_of_elements) == 0:
        return -1
    return max(set(list_of_elements), key=list_of_elements.count)


def project_3d_to_2d(joints_3d, bbox, pred_cam):
    joints_3d_tensor = torch.from_numpy(joints_3d).unsqueeze(0)
    focal_length = 5000
    camera_center = torch.zeros(2).unsqueeze(0)
    rotation = torch.eye(3).unsqueeze(0)
    pred_camera = torch.from_numpy(pred_cam).unsqueeze(0)  # shape: (1, 3)
    pred_cam_t = torch.stack(
        [
            pred_camera[:, 1],
            pred_camera[:, 2],
            2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)
        ],
        dim=-1
    )

    joints_2d = perspective_projection(
        joints_3d_tensor,
        rotation,
        pred_cam_t,
        focal_length,
        camera_center
    ).squeeze().numpy()

    # scale back to the original image
    scaled_joints_2d = joints_2d * bbox[2:] / 224 + bbox[:2]
    scaled_joints_2d = scaled_joints_2d.astype(int)
    return scaled_joints_2d


def find_min_dist_and_angle(bb, dr, cw_feats, norm_factor=10):
    if len(cw_feats) < 1:
        return 0, 0
    dist_angle = [(np.linalg.norm(bb[:2] - cwf[0]), angle_between(dr, cwf[1].reshape(2))) for cwf in cw_feats]
    dist_min, angle_min = min(dist_angle, key=lambda x: x[0])
    dist_min = np.exp(-dist_min * 1.7 / (bb[3] * norm_factor))
    angle_min = 1 - np.cos(angle_min * np.pi / 180)
    return dist_min, angle_min


def find_min_dist_and_angle_veh(bb, dr, veh_feats, norm_factor=10):
    if len(veh_feats) < 1:
        return 0, 0, 0
    dist_angle_speed = [
        (np.linalg.norm(bb[:2] - veh[0]), angle_between(dr, veh[1].reshape(2)), veh[2]) for veh in veh_feats]
    dist_min, angle_min, speed_min = min(dist_angle_speed, key=lambda x: x[0])
    dist_min = np.exp(-dist_min * 1.5 / (bb[3] * norm_factor))
    angle_min = 1 - np.cos(angle_min * np.pi / 180)
    speed_min = speed_min[0] if speed_min >= 0.05 else 0
    return dist_min, angle_min, speed_min


def extract_v2p_and_env(ped_to_analyze, ped_results):
    for ped_id in ped_to_analyze:
        num_group, d_veh, a_veh, s_veh, d_cw, a_cw, sem = [], [], [], [], [], [], []
        for fr, bb, jo, pc, dr in zip(
            ped_results[ped_id]['frames'],
            ped_results[ped_id]['bboxes'],
            ped_results[ped_id]['joints_3d'],
            ped_results[ped_id]['pred_cam'],
            ped_results[ped_id]['direction']
            ):
            # number of group
            temp_num = 0
            for ped_temp in set(ped_to_analyze) - set([ped_id]):
                f_idx = np.where(ped_results[ped_temp]['frames'] == fr)[0]
                if f_idx.size != 0 and np.linalg.norm(bb[:2] - ped_results[ped_temp]['bboxes'][f_idx][0, :2]) * 1.7 / bb[3] < 5:
                    temp_num += 1
            num_group.append(temp_num)

            # v2p interaction
            temp_veh_feats, temp_veh_speed = [], []
            for v_id in frames_ped_veh[fr]['veh_id']:
                f_idx = np.where(veh[v_id]['frames'] == fr)[0]
                v_bb, v_dr, v_speed = veh[v_id]['bboxes'][f_idx][0, :2], veh[v_id]['direction'][f_idx], veh[v_id]['speed'][f_idx]
                temp_veh_feats.append((v_bb, v_dr, v_speed))
            
            dist_min, angle_min, speed_min = find_min_dist_and_angle_veh(bb, dr, temp_veh_feats)
            d_veh.append(dist_min)
            a_veh.append(angle_min)
            s_veh.append(speed_min)

            # crosswalk context: distance & angle
            dist_min, angle_min = find_min_dist_and_angle(bb, dr, crosswalk_features[video_id])
            d_cw.append(dist_min)
            a_cw.append(angle_min)
            
            # location context: semantic label of the current position
            heel_l_r = project_3d_to_2d(jo, bb, pc)[[21, 24]]  # left and right heel
            sem_ped = [
                semantics[heel_l_r[k][1]+i, heel_l_r[k][0]+j] for i in range(-1, 2) for j in range(-1, 2) for k in range(2) if 0<=heel_l_r[k][1]+i<IMG_HEIGHT and 0<=heel_l_r[k][0]+j<IMG_WIDTH
            ]
            sem.append(most_frequent(sem_ped))
        ped_results[ped_id]['num_group'] = np.array(num_group)
        ped_results[ped_id]['v2p_dist'] = np.array(d_veh)
        ped_results[ped_id]['v2p_angle'] = np.array(a_veh)
        ped_results[ped_id]['v2p_speed'] = np.array(s_veh)
        ped_results[ped_id]['env_loc'] = np.array(sem)
        ped_results[ped_id]['env_cw_dist'] = np.array(d_cw)
        ped_results[ped_id]['env_cw_angle'] = np.array(a_cw)


def update_dicts(list_of_same_ids, detection_results, frames_ped_veh, height_factor=1.7, norm_factor=5, treat_cam=True):
    keys_to_deal = ['frames', 'bboxes', 'joints_3d']
    if treat_cam:
        keys_to_deal.append('pred_cam')
    ref_id = list_of_same_ids[0]
    for one_id in list_of_same_ids[1:]:
        for one_frame in detection_results[one_id]['frames']:
            frames_ped_veh[one_frame]['ped_id'].add(ref_id)
            frames_ped_veh[one_frame]['ped_id'].remove(one_id)
        for one_key in keys_to_deal:
            detection_results[ref_id][one_key] = np.concatenate((
                detection_results[ref_id][one_key], detection_results[one_id][one_key]))
        del detection_results[one_id]
    
    speed, direction = [], []
    fr_prev, bbox_prev = detection_results[ref_id]['frames'][0], detection_results[ref_id]['bboxes'][0]
    for fr, bbox in zip(detection_results[ref_id]['frames'], detection_results[ref_id]['bboxes']):
        direction_temp = bbox[:2] - bbox_prev[:2]
        speed_temp = np.linalg.norm(
            direction_temp) * 60 * height_factor / ((bbox[3] + bbox_prev[3]) * (fr - fr_prev) * norm_factor) if fr != fr_prev else 0
        speed.append(speed_temp)
        direction.append(direction_temp)
        fr_prev, bbox_prev = fr, bbox
    detection_results[ref_id]['speed'] = np.array(speed)
    detection_results[ref_id]['direction'] = np.array(direction)


def save_csv(ped_to_analyze, ped_results):
    joints_selected = np.array(list(set(range(2, 17)) - {8}))
    for ped_id in ped_to_analyze:
        length_of_frames = len(ped_results[ped_id]['frames'])
        
        v_id      = pd.DataFrame([video_id] * length_of_frames)
        p_id      = pd.DataFrame([ped_id] * length_of_frames)
        frames    = pd.DataFrame(ped_results[ped_id]['frames'])
        pose_feat = pd.DataFrame(ped_results[ped_id]['joints_3d'][:, joints_selected].reshape(-1, 42))
        n_group   = pd.DataFrame(ped_results[ped_id]['num_group'])
        p_speed   = pd.DataFrame(ped_results[ped_id]['speed'])

        v_dist    = pd.DataFrame(ped_results[ped_id]['v2p_dist'])
        v_angle   = pd.DataFrame(ped_results[ped_id]['v2p_angle'])
        v_speed   = pd.DataFrame(ped_results[ped_id]['v2p_speed'])

        e_dist    = pd.DataFrame(ped_results[ped_id]['env_cw_dist'])
        e_angle   = pd.DataFrame(ped_results[ped_id]['env_cw_angle'])
        e_loc     = pd.DataFrame(ped_results[ped_id]['env_loc'])
        result = pd.concat([
            v_id, p_id, frames,
            pose_feat, n_group, p_speed,
            v_dist, v_angle, v_speed,
            e_dist, e_angle, e_loc], axis=1)
        result.columns = ['video', 'ped_id', 'frame'] + \
            ['pose_{}_{}'.format(num, ax) for num in range(14) for ax in ['x', 'y', 'z']] + \
            ['n_group', 'p_speed', 'v_dist', 'v_angle', 'v_speed', 'e_dist', 'e_angle', 'e_loc']
        result.to_csv('V{}_P{}.csv'.format(video_id, ped_id))


def update_all_and_save(list_of_list):
    for l in list_of_list:
        update_dicts(l, ped_results, frames_ped_veh, height_factor=1.7, treat_cam=True)
    for v in veh.keys():
        update_dicts([v], veh, frames_ped_veh, height_factor=1.5, norm_factor=10, treat_cam=False)
    extract_v2p_and_env([l[0] for l in list_of_list], ped_results)
    save_csv([l[0] for l in list_of_list], ped_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        help='root folder containing images from videos')
    parser.add_argument('--output_dir', type=str,
                        help='output folder to save detection/tracking results')
    args = parser.parse_args()
    
    cw_pos = ((100, 719), (900, 600))
    center = np.array([cw_pos[0][0]+cw_pos[1][0], cw_pos[0][1]+cw_pos[1][1]])
    direct = np.array([cw_pos[0][0]-cw_pos[1][0], cw_pos[0][1]-cw_pos[1][1]])
    crosswalk_features = (center, direct)

    # ========= Start Feature Extraction ========= #
    sub_dirs = os.listdir(args.root_dir)
    for sub_dir in sub_dirs:
        if not os.path.isdir(os.path.join(args.root_dir, sub_dir)):
            continue
        video_dirs = glob.glob(os.path.join(args.root_dir, sub_dir, '*/'))
        for video_dir in video_dirs:
            filename = os.path.join(video_dir, 'detection.pickle')
            with open(filename, 'rb') as handle:
                detections = pickle.load(handle)
            base_name = video_dir.split('/')[-2]
            annotation_pedest = pd.read_excel(os.path.join('data_annotation', base_name + '_pedestrian.xlsx'))
            annotation_signal = pd.read_excel(os.path.join('data_annotation', base_name + '_signal.xlsx'))
            import pdb; pdb.set_trace()

    '''
    update_all_and_save([[1, 20, 25], [2, 23]])
    '''