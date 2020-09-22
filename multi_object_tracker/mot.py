import os
import cv2
import time
import torch
import shutil
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from yolov3.yolo import YOLOv3

from multi_object_tracker import Sort, iou
from multi_object_tracker.data import ImageFolder, images_to_video

from scipy.optimize import linear_sum_assignment
import pdb, traceback, sys, code


ind_to_class = {0: 'person', 1: 'bicycle', 2: 'car', 5: 'bus', 7: 'truck'}


class MOT():
    def __init__(
            self,
            device=None,
            batch_size=12,
            display=False,
            detection_threshold=0.7,
            detector_type='yolo',
            yolo_img_size=608,
            output_format='list',
    ):
        '''
        Multi Person Tracker

        :param device (str, 'cuda' or 'cpu'): torch device for model and inputs
        :param batch_size (int): batch size for detection model
        :param display (bool): display the results of multi person tracking
        :param detection_threshold (float): threshold to filter detector predictions
        :param detector_type (str, 'maskrcnn' or 'yolo'): detector architecture
        :param yolo_img_size (int): yolo detector input image size
        :param output_format (str, 'dict' or 'list'): result output format
        '''

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = batch_size
        self.display = display
        self.detection_threshold = detection_threshold
        self.output_format = output_format

        if detector_type == 'maskrcnn':
            self.detector = keypointrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        elif detector_type == 'yolo':
            self.detector = YOLOv3(
                device=self.device, img_size=yolo_img_size, person_detector=False, video=True, return_dict=True
            )
        else:
            raise ModuleNotFoundError

        self.tracker = {'people': Sort(), 'vehicles': Sort()}

    @torch.no_grad()
    def run_tracker(self, dataloader):
        '''
        Run tracker on an input video

        :param video (ndarray): input video tensor of shape NxHxWxC. Preferable use skvideo to read videos
        :return: trackers (ndarray): output tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        '''

        # initialize tracker
        # self.tracker = Sort()

        start = time.time()
        print('Running Multi-Object-Tracker')
        trackers_people, trackers_vehicles = [], []
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            predictions = self.detector(batch)

            for pred in predictions:
                bb = pred['boxes'].cpu().numpy()
                sc = pred['scores'].cpu().numpy()[..., None]
                cc = pred['classes'].cpu().numpy()
                
                dets = np.hstack([bb,sc])
                dets = dets[sc[:,0] > self.detection_threshold]
                cc = cc[sc[:,0] > self.detection_threshold]

                dets_people, dets_vehicles = dets[cc==0], dets[np.logical_or.reduce((cc==1, cc==2, cc==5, cc==7))]
                self.update_trackers(dets_people, self.tracker["people"], trackers_people)
                self.update_trackers(dets_vehicles, self.tracker["vehicles"], trackers_vehicles, cc[np.logical_or.reduce((cc==1, cc==2, cc==5, cc==7))])
                
                '''
                # if nothing detected do not update the tracker
                if dets.shape[0] > 0:
                    track_bbs_ids = self.tracker.update(dets)
                else:
                    track_bbs_ids = np.empty((0, 5))
                import pdb; pdb.set_trace()
                trackers.append(track_bbs_ids)
                '''
        runtime = time.time() - start
        fps = len(dataloader.dataset) / runtime
        print(f'Finished. Detection + Tracking FPS {fps:.2f}')
        return trackers_people, trackers_vehicles
    
    @staticmethod
    def update_trackers(dets, tracker, list_of_results, list_class=None):
        if dets.shape[0] > 0:
            track_bbs_ids = tracker.update(dets)

            if list_class is not None and track_bbs_ids.size != 0:
                iou_matrix = np.zeros((len(dets), len(dets)), dtype=np.float32)
                for idx_d, d in enumerate(dets):
                    for idx_t, tbi in enumerate(track_bbs_ids):
                        iou_matrix[idx_d, idx_t] = iou(d[:4], tbi[:4])
                _, matched_idx = linear_sum_assignment(-iou_matrix)
                # import pdb; pdb.set_trace()
                track_bbs_ids = np.concatenate((track_bbs_ids, np.array(list_class[matched_idx[:track_bbs_ids.shape[0]]]).reshape(-1, 1)), axis=1)
        else:
            track_bbs_ids = np.empty((0, 5))
        if track_bbs_ids.size == 0:
            track_bbs_ids = np.empty((0, 5)) if list_class is None else np.empty((0, 6))
        list_of_results.append(track_bbs_ids)

    def prepare_output_tracks_people(self, trackers):
        '''
        Put results into a dictionary consists of detected people
        :param trackers (ndarray): input tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        :return: dict: of people. each key represent single person with detected bboxes and frame_ids
        '''
        people = dict()

        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                person_id = int(d[4])
                # bbox = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]]) # x1, y1, w, h

                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w/2, d[1] + h/2
                # w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox' : [],
                        'frames' : [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
        for k in people.keys():
            people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 4))
            people[k]['frames'] = np.array(people[k]['frames'])

        return people

    def prepare_output_tracks_vehicles(self, trackers):
        '''
        Put results into a dictionary consists of detected vehicles
        :param trackers (ndarray): input tracklets of shape Nx6 [x1,y1,x2,y2,track_id,class]
        :return: dict: of vehicles. each key represent single vehicle with detected bboxes, frame_ids, classes
        '''
        vehicles = dict()

        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                vehicle_id = int(d[4])
                vehicle_class = int(d[5])
                # bbox = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]]) # x1, y1, w, h

                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w/2, d[1] + h/2
                # w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])

                if vehicle_id in vehicles.keys():
                    vehicles[vehicle_id]['bboxes'].append(bbox)
                    vehicles[vehicle_id]['frames'].append(frame_idx)
                    vehicles[vehicle_id]['classes'].append(vehicle_class)
                else:
                    vehicles[vehicle_id] = {
                        'bboxes' : [],
                        'frames' : [],
                        'classes' : [],
                    }
                    vehicles[vehicle_id]['bboxes'].append(bbox)
                    vehicles[vehicle_id]['frames'].append(frame_idx)
                    vehicles[vehicle_id]['classes'].append(vehicle_class)
        for k in vehicles.keys():
            vehicles[k]['bboxes'] = np.array(vehicles[k]['bboxes']).reshape((len(vehicles[k]['bboxes']), 4))
            vehicles[k]['frames'] = np.array(vehicles[k]['frames'])
            vehicles[k]['classes'] = np.array(vehicles[k]['classes'])

        return vehicles

    def display_results(self, image_folder, trackers_people, trackers_vehicles, output_folder=None):
        '''
        Display the output of multi-person-tracking
        :param video (ndarray): input video tensor of shape NxHxWxC
        :param trackers (ndarray): tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        :return: None
        '''
        print('Displaying results..')

        save = True if output_folder else False
        tmp_write_folder = output_folder  # osp.join('/tmp', f'{osp.basename(image_folder)}_mot_results')
        os.makedirs(tmp_write_folder, exist_ok=True)

        colours = np.random.rand(32, 3)
        image_file_names = sorted([
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for idx, (img_fname, tracker_p, tracker_v) in enumerate(zip(image_file_names, trackers_people, trackers_vehicles)):

            img = cv2.imread(img_fname)
            for d in tracker_p:
                d = d.astype(np.int32)
                c = (colours[d[4] % 32, :] * 255).astype(np.uint8).tolist()
                cv2.rectangle(
                    img, (d[0], d[1]), (d[2], d[3]),
                    color=c, thickness=int(round(img.shape[0] / 256))
                )
                cv2.putText(img, f'P: {d[4]}', (d[0] - 9, d[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(img, f'P: {d[4]}', (d[0] - 8, d[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
            for d in tracker_v:
                d = d.astype(np.int32)
                c = (colours[d[4] % 32, :] * 255).astype(np.uint8).tolist()
                cv2.rectangle(
                    img, (d[0], d[1]), (d[2], d[3]),
                    color=c, thickness=int(round(img.shape[0] / 256))
                )
                cv2.putText(img, f'V[{ind_to_class[d[5]]}]: {d[4]}', (d[0] - 9, d[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(img, f'V[{ind_to_class[d[5]]}]: {d[4]}', (d[0] - 8, d[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(img, f'frame number: {idx}', (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255))
            cv2.imshow('result video', img)

            # time.sleep(0.03)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if save:
                cv2.imwrite(osp.join(tmp_write_folder, f'{idx:06d}.jpg'), img)

        cv2.destroyAllWindows()
        '''
        if save:
            print(f'Saving output video to {output_folder}')
            images_to_video(img_folder=tmp_write_folder, output_vid_file=output_folder)
            shutil.rmtree(tmp_write_folder)
        '''

    def __call__(self, image_folder, output_folder=None):
        '''
        Execute MOT and return results as a dictionary of person instances

        :param video (ndarray): input video tensor of shape NxHxWxC
        :return: a dictionary of person instances
        '''

        image_dataset = ImageFolder(image_folder)

        dataloader = DataLoader(image_dataset, batch_size=self.batch_size, num_workers=8)

        trackers_people, trackers_vehicles = self.run_tracker(dataloader)
        if self.display:
            self.display_results(image_folder, trackers_people, trackers_vehicles, output_folder)

        if self.output_format == 'dict':
            result_people = self.prepare_output_tracks_people(trackers_people)
            result_vehicles = self.prepare_output_tracks_vehicles(trackers_vehicles)
        elif self.output_format == 'list':
            result_people, result_vehicles = trackers_people, trackers_vehicles

        return result_people, result_vehicles
