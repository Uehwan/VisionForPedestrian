import os
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset


VIDEO_FPS = 30
FRAME_MARGIN = 3


class PedestrianDataset(Dataset):
    def __init__(
        self,
        data_path="./data_raw/train",
        context_len=1,
        future_stamps=[0.5, 1.0, 1.5],
        transformer_input=False,
        state_info=True
    ):
        self.data_path = data_path
        self.context_len = context_len
        self.future_stamps = future_stamps
        self.transformer_input = transformer_input
        self.state_info = state_info
        self.num_to_sample = int(self.context_len * VIDEO_FPS / 2)
        self.instances, self.stamps = self.gather_data_instances()
        print("Data Instances Gathereing Completed!")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx], self.stamps[idx]

    def gather_data_instances(self):
        instances, stamps = [], []
        csv_files = os.listdir(self.data_path)
        for csv_file in csv_files:
            one_csv = pd.read_csv(os.path.join(self.data_path, csv_file))
            f_start = None
            temp_inst, temp_stamp = [], [[] for _ in range(len(self.future_stamps))]
            for idx in range(len(one_csv)):
                curr_row = one_csv.iloc[idx]
                if f_start is None:
                    f_start = int(curr_row['frame'])
                    f_boundary = f_start + self.context_len * VIDEO_FPS + FRAME_MARGIN
                    v_range = [int(f_start + (self.context_len + f_stamp) * VIDEO_FPS) for f_stamp in self.future_stamps]
                    stamp_ranges = [range(vr-FRAME_MARGIN, vr+FRAME_MARGIN) for vr in v_range]
                
                if int(curr_row['frame']) < f_boundary:
                    # gather instances within the time context
                    curr_vec = np.nan_to_num(curr_row.values[4:])
                    curr_vec[-2] /= 50
                    if not self.state_info:
                        curr_vec = curr_vec[:-1]
                    if self.transformer_input:
                        curr_vec = np.concatenate((curr_vec, np.zeros(52-curr_vec.shape[0])))
                    temp_inst.append(curr_vec)
                else:
                    # now the buffer is full, let's retrieve a label stamp
                    if len(temp_inst) >= self.num_to_sample:    
                        # enough context as an instance
                        temp_idx = idx + 1
                        while(temp_idx < len(one_csv) and temp_idx < max(stamp_ranges[-1])):
                            temp_row = one_csv.iloc[temp_idx]
                            temp_frame = temp_row['frame']
                            check_inclusion = [temp_frame in sr for sr in stamp_ranges]
                            for i, ck in enumerate(check_inclusion):
                                if ck: temp_stamp[i].append(temp_idx)
                            temp_idx += 1
                        if all([len(ts) > 0 for ts in temp_stamp]):
                            instances.append(np.array(random.sample(temp_inst, self.num_to_sample)))
                            temp_indices = [random.sample(ts, 1)[0] for ts in temp_stamp]
                            temp_stamp = [one_csv.iloc[ti]['crossing'] for ti in temp_indices]
                            stamps.append(np.array(temp_stamp))

                    # reinitialize and get ready for another instance
                    f_start = None
                    temp_inst, temp_stamp = [], [[] for _ in range(len(self.future_stamps))]
        return instances, stamps


def build_dataset(image_set, args):
    data_path = os.path.join(args.data_path, image_set)
    ds = PedestrianDataset(
        data_path=data_path,
        context_len=args.context_length,
        future_stamps=args.future_stamps,
        transformer_input=args.model_type=='transformer',
        state_info=args.state_info
    )
    return ds


if __name__ == "__main__":
    ds = PedestrianDataset(future_stamps=[1.0])
    sample_inst, sample_stamp = ds[0]
    print("total instances:", len(ds))
    print("shape of sample_inst:", sample_inst.shape)
    print("shape of sample_stamp:", sample_stamp.shape)
