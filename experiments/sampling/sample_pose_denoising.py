import sys
sys.path.append('')
import torch
import os
import matplotlib.pyplot as plt

from FlowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from VectorFieldModels.Transformer import DiT
from VectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero

from Utils.NRDF.utils.transforms import quaternion_to_axis_angle
from Utils.NRDF.utils.data_utils import amass_splits
import pytorch3d.transforms as transforms
import glob

import numpy as np

def noise_pose(clean_pose, noise_level=0.1):
    noise = np.random.randn(*clean_pose.shape)
    noisy_pose = clean_pose + noise * noise_level

    return noisy_pose



def gen_noisy(args):
    clean_seqs = sorted(glob.glob(args['directory'] + args['clean'] + '/*/*.npz'))
    clean_seqs = [ds for ds in clean_seqs if ds.split('/')[-2] in amass_splits['test']]


    for k in range(args['no_samples']):
        clean_pose_samples = []
        noisy_pose_samples = []
        for i, seq in enumerate(clean_seqs):
            cdata = np.load(seq)
            clean_poses = cdata['pose_body'][:, :63].astype(np.float32)
            clean_poses = clean_poses.reshape(-1, 21, 3)
            noisy_poses = noise_pose(clean_poses, noise_level=args['noise_level'])
            
            clean_pose_samples.append(clean_poses)
            noisy_pose_samples.append(noisy_poses)
            print("done" + seq)
        
        clean_pose_samples = np.concatenate(clean_pose_samples, axis=0)
        noisy_pose_samples = np.concatenate(noisy_pose_samples, axis=0)
        np.savez(args['directory'] + args['save_location'] + f'clean/{k}.npz', pose_body=clean_pose_samples)
        np.savez(args['directory'] + args['save_location'] + f'noisey/{k}.npz', pose_body=noisy_pose_samples)
    

if __name__ == '__main__':
    args = {
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'clean': './dataset/amass/SAMPLED_POSES/',

        'save_location': 'experiments/samples/denoised_pose/', 
        'noise_level': 0.2,
        'no_samples': 20,
    }
    gen_noisy(args)


