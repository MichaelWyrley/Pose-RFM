# Taken from https://github.com/nghorbani/amass/blob/a9888a92a4e62533454aa43e5f979d9a8bc8c893/notebooks/01-AMASS_Visualization.ipynb
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
# add the current working directory so this can be run from the github repo root !!
sys.path.append(os.getcwd())
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from scipy.linalg import sqrtm

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp
from human_body_prior.body_model.body_model import BodyModel
from Utils.NRDF.utils.data_utils import amass_splits
import pytorch3d.transforms as transforms


def pose_to_vert(pose_body, args, device='cuda'):

    bm_fname = osp.join(args['directory'], args['model'])

    num_betas = 16 # number of body parameters

    bm = BodyModel(bm_fname, num_betas=num_betas, model_type='smplh').to(device)
    
    time_length = len(pose_body)
  
    body_pose_beta = bm(pose_body=torch.Tensor(pose_body.reshape(time_length, -1)).to(device))

    return body_pose_beta.Jtr

def dataset_frechet_distance(args, device='cuda'):

    dataset_seq = sorted(glob.glob(args['directory'] + args['dataset_directory'] + '/*/*.npz'))
    dataset_seq = [ds for ds in dataset_seq if ds.split('/')[-2] in amass_splits['test']]

    dataset_mean = np.zeros((len(dataset_seq), 22*3))
    dataset_std = np.zeros((len(dataset_seq), 22*3, 22*3 ))

    for i in range(len(dataset_seq)):
        dataset = np.load(dataset_seq[i])

        batched_poses = np.array_split(dataset['pose_body'], args['dataset_size'])

        batched_mean = np.zeros((len(batched_poses), 22*3))
        batched_std = np.zeros((len(batched_poses), 22*3, 22*3 ))
                
        for j, pose in enumerate(batched_poses):
            dataset_pose = pose[:, :63]
            dataset_pose = pose_to_vert(dataset_pose, args)
            dataset_pose = dataset_pose[:, :22].reshape(-1, 66).detach().cpu().numpy()

            batched_mean[j] = np.mean(dataset_pose, axis=0)
            batched_std[j] = np.cov(dataset_pose,rowvar=False)

        dataset_mean[i] = batched_mean.mean(0)
        dataset_std[i] = batched_std.mean(0)

    dataset_mean = dataset_mean.mean(0)
    dataset_std = dataset_std.mean(0)

    
    np.savez(args['directory'] + args['generated_directory'] + f'dataset_mean_cov.npz', mu=dataset_mean, std=dataset_std)
    
    return dataset_mean, dataset_std


if __name__ == '__main__':
    args = {
        # 'support_dir': '/vol/bitbucket/mew23/individual_project/',
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'dataset_directory': './dataset/amass/SAMPLED_POSES/',
        'generated_directory': 'experiments/utils/',
        'model': './dataset/models/neutral/model.npz',

        'dataset_name': 'pose',

        'dataset_size': 500,
        

    }
    print(dataset_frechet_distance(args))
