import csv
import os
import argparse
import torch
import torchvision
import numpy as np
import pandas as pd
import scipy
import PIL.Image
import pickle

from tqdm import tqdm
from torch.utils.data import DataLoader

from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

from multi_object_tracker import MOT
from lib.models.vibe import VIBE_Demo
from lib.models.spin import perspective_projection
from lib.dataset.inference import Inference
from lib.utils.demo_utils import (
    download_segmentation,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

from demo_on_single_images import angle_between


def most_frequent(list_of_elements):
    if len(list_of_elements) == 0:
        return -1
    return max(set(list_of_elements), key=list_of_elements.count)


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


def semantic_segmentation(image_folder):
    download_segmentation()
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet101dilated',
        fc_dim=2048,
        weights='data/segm_data/encoder_epoch_25.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='data/segm_data/decoder_epoch_25.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()

    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    pil_image = PIL.Image.open(os.path.join(image_folder, '000000.jpg')).convert('RGB')
    img_original = np.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
        
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    del net_encoder, net_decoder, segmentation_module
    return pred


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


def save_csv(ped_to_analyze, ped_results):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')
    parser.add_argument('--vid_id', type=str,
                        help='video id to distinguish videos')
    parser.add_argument('--img_height', type=int, default=1080,
                        help='image height (video)')
    parser.add_argument('--img_width', type=int, default=1920,
                        help='image width (video)')
    parser.add_argument('--output_folder', type=str,
                        help='output folder name to save detection/tracking results')
    
    args = parser.parse_args()

    video_file = args.vid_file
    vid_id = args.vid_id
    IMG_WIDTH, IMG_HEIGHT = args.img_width, args.img_height

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # ========= Define Multi-Object tracker ========= #
    mot = MOT(
        device=device,
        batch_size=12,
        display=True,
        detector_type='yolo',
        output_format='dict',
        yolo_img_size=608
    )

    bbox_scale = 1.1
    joints_selected = np.array(list(set(range(2, 17)) - {8}))
    
    image_folder, _, _ = video_to_images(video_file, img_folder='./vid_to_img')
    output_folder = args.output_folder
    ped_results = {}
    frames_ped_veh = {}
    ped, veh = mot(image_folder, output_folder=output_folder)

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
    
    for person_id in tqdm(sorted(list(ped.keys()))):
        bboxes = joints2d = None

        bboxes = ped[person_id]['bbox']
        frames = ped[person_id]['frames']

        for frame_one in frames:
            if frame_one not in frames_ped_veh:
                frames_ped_veh[frame_one] = {
                    'ped_id': set(),
                    'veh_id': set()
                }
            frames_ped_veh[frame_one]['ped_id'].add(person_id)

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=450, num_workers=16)

        with torch.no_grad():
            pred_joints3d, norm_joints2d, pred_cam = [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))
                batch = batch.unsqueeze(0)
                batch = batch.to(device)
                batch_size, seqlen = batch.shape[:2]
                
                output = model(batch)[-1]
                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            pred_cam = torch.cat(pred_cam, dim=0)
            del batch
            pred_joints3d = pred_joints3d.cpu().numpy()
            pred_cam = pred_cam.cpu().numpy()
            ped_results[person_id] = {
                'joints_3d': pred_joints3d,
                'bboxes': bboxes,
                'frames': frames,
                'pred_cam': pred_cam
            }
    
    del model
    
    for veh_id in veh:
        frames = veh[veh_id]['frames']

        for frame_one in frames:
            if frame_one not in frames_ped_veh:
                frames_ped_veh[frame_one] = {
                    'ped_id': set(),
                    'veh_id': set()
                }
            frames_ped_veh[frame_one]['veh_id'].add(veh_id)    

    ########################SEMANTIC SEGMENTATION##########################
    names = {}
    with open('data/segm_data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    
    semantics = semantic_segmentation(image_folder)

    with open('anno_crosswalk.pickle', 'rb') as pickle_file:
        crosswalk_positions = pickle.load(pickle_file)
    
    crosswalk_features = {}
    for vid_id in crosswalk_positions.keys():
        cw_feat = []
        cw_pos = crosswalk_positions[vid_id]
        for i in range(0, len(cw_pos), 2):
            center = np.array([cw_pos[i][0]+cw_pos[i+1][0], cw_pos[i][1]+cw_pos[i+1][1]])
            direct = np.array([cw_pos[i][0]-cw_pos[i+1][0], cw_pos[i][1]-cw_pos[i+1][1]])
            cw_feat.append((center, direct))
        crosswalk_features[vid_id] = cw_feat
    
    
    def update_all_and_save(list_of_list):
        for l in list_of_list:
            update_dicts(l, ped_results, frames_ped_veh, height_factor=1.7, treat_cam=True)
        for v in veh.keys():
            update_dicts([v], veh, frames_ped_veh, height_factor=1.5, norm_factor=10, treat_cam=False)
        extract_v2p_and_env([l[0] for l in list_of_list], ped_results)
        save_csv([l[0] for l in list_of_list], ped_results)
    
    '''
    update_all_and_save([[1, 20, 25], [2, 23]])
    '''
