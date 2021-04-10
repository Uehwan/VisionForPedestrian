import os
import glob
import pickle
import ntpath
import argparse
import PIL.Image
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader

from mit_semseg.models import ModelBuilder, SegmentationModule
from multi_object_tracker import MOT
from lib.models.vibe import VIBE_Demo
from lib.dataset.inference import Inference
from lib.utils.demo_utils import (
    download_segmentation,
    download_ckpt,
)


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
    pil_image = PIL.Image.open(os.path.join(image_folder, '000001.png')).convert('RGB')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        help='root folder containing images from videos')
    parser.add_argument('--output_dir', type=str,
                        help='output folder to save detection/tracking results')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bbox_scale = 1.1

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights for VIBE ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Start Detection and Tracking ========= #
    sub_dirs = os.listdir(args.root_dir)
    for sub_dir in sub_dirs:
        video_dirs = glob.glob(os.path.join(args.root_dir, sub_dir, '*/'))
        for image_folder in video_dirs:
            output_folder = os.path.join(
                args.output_dir, sub_dir, ntpath.basename(image_folder))
            ped_results, frames_ped_veh = {}, {}
            
            # ========= Define Multi-Object tracker ========= #
            mot = MOT(
                device=device,
                batch_size=12,
                display=True,
                detector_type='yolo',
                output_format='dict',
                yolo_img_size=608
            )
            
            ########################Detection and Tracking##########################
            ped, veh = mot(image_folder, output_folder=output_folder)
            
            ########################3D Human Pose Estimation##########################
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
            semantics = semantic_segmentation(image_folder)
            to_store = {
                'ped': ped,
                'veh': veh,
                'ped_results': ped_results,
                'frames_ped_veh': frames_ped_veh,
                'semantics': semantics
            }
            file_name = os.path.join(output_folder, 'detection.pickle')
            with open(file_name, 'wb') as handle:
                pickle.dump(to_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
