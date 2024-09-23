# taken from https://github.com/hynann/NRDF/blob/master
import sys
sys.path.append('')

import os
import numpy as np
import torch

from torch.utils.data import Dataset

from main.utils.NRDF.utils.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
from main.utils.NRDF.utils.data_utils import quaternion_hamilton_product, amass_splits

class PoseData(Dataset):
    def __init__(self, mode, clean_dir, batch_size=4, num_workers=6, stage=1, flip=False,
                 random_poses=False):
        self.mode = mode
        self.clean_dir = clean_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flip = flip
        self.stage = stage

        self.clean_data = np.load(self.clean_dir + f'{mode}.npz')
        self.poses = torch.from_numpy(self.clean_data['pose_body']).to(torch.float32)
        self.categories = torch.from_numpy(self.clean_data['categories']).to(int)
        
    
    def __len__(self):
        return len(self.clean_data['pose_body'])

    def __getitem__(self, idx):
        return (self.poses[idx], self.categories[idx])
    
    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size,  shuffle=shuffle,
            worker_init_fn=self.worker_init_fn, drop_last=True)
    
    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

