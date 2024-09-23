import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
# add the current working directory so this can be run from the github repo root !!
sys.path.append(os.getcwd())
import torch
import numpy as np
import glob
from scipy.linalg import sqrtm

from os import path as osp
from human_body_prior.body_model.body_model import BodyModel


from main.utils.NRDF.data.gen_data import nn_search
from main.utils.NRDF.utils.data_utils import geo, load_faiss
from main.utils.NRDF.utils.transforms import axis_angle_to_quaternion


def pose_to_vert(pose_body, args, device='cuda'):

    num_betas = 16 # number of body parameters

    bm = BodyModel(args['model'], num_betas=num_betas, model_type='smplh').to(device)
    
    time_length = len(pose_body)
  
    body_pose_beta = bm(pose_body=torch.Tensor(pose_body.reshape(time_length, -1)).to(device))

    return body_pose_beta.Jtr


def average_pairwise_distance(args):

    gen_seq = sorted(glob.glob(args['generated_directory'] + '*.npz'))

    pose_dist = []
    
    for gen_dir in gen_seq:
        print("starting " + gen_dir.split('/')[-1].split('.')[0])
        bdata = np.load(gen_dir)
        pose_body = bdata[args['dataset_name']]
        pose_body = pose_to_vert(pose_body, args)[:, :22]

        num_frames = pose_body.shape[0]
        num_joints = pose_body.shape[1]

        total_distance = 0
        total_sum = 0
        for i in range(num_frames):
            for j in range(i+1, num_frames):
                total_distance += torch.mean(torch.linalg.norm(pose_body[i] - pose_body[j], dim = 1)).item()
                total_sum += 1

        pose_dist.append(total_distance / total_sum)


    return np.mean(pose_dist), np.std(pose_dist)

def distance_gen_dataset(args, device='cuda'):
    # Calcualte the distances between each generated sample and its coresponding minimum dataset sample

    gen_seq = sorted(glob.glob(args['generated_directory'] + '*.npz'))
    pose_dist = []

    # load faiss related terms
    index, all_poses_aa, all_poses_quat = load_faiss(args['faiss_model'])

    for gen_dir in gen_seq:
        print("starting " + gen_dir.split('/')[-1].split('.')[0])
        bdata = np.load(gen_dir)
        pose_body = bdata[args['dataset_name']].reshape(-1, 21, 3)
        
        pose_body_quat = axis_angle_to_quaternion(torch.from_numpy(pose_body)).detach().cpu().numpy() # [bs, nj, 4]
        dist_cal = geo()

        # search nearest neighbors
        _, _, dist_gt = nn_search(quat=pose_body_quat, 
                                                index=index, 
                                                dist_cal=dist_cal, 
                                                all_poses_aa=all_poses_aa, 
                                                all_poses_quat=all_poses_quat, 
                                                k_faiss=args['k-faiss'], 
                                                k_dist=args['k-dist'])
        
        pose_dist.append(dist_gt.mean())

        
    pose_dist = np.array(pose_dist)
    return np.mean(pose_dist), np.std(pose_dist)

def frechet_distance(args, device='cuda'):

    gen_seq = sorted(glob.glob(args['generated_directory'] + '*.npz'))

    pose_dist = []

    cdata = np.load(args['dataset_data'])
    dataset_mu, dataset_std = cdata['mu'], cdata['std']
    
    for gen_dir in gen_seq:
        print("starting " + gen_dir.split('/')[-1].split('.')[0])

        bdata = np.load(gen_dir)
        gen_pose = bdata[args['dataset_name']].reshape(-1, 21 * 3)

        gen_pose = pose_to_vert(gen_pose, args)
        gen_pose = gen_pose[:, :22].reshape(-1, 66).detach().cpu().numpy()

        mean, std = np.mean(gen_pose, axis=0), np.cov(gen_pose, rowvar=False)

        ssdiff = np.sum((mean - dataset_mu)**2)
        covmean, _ = sqrtm(np.dot(std, dataset_std), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        pose_dist.append(ssdiff + np.trace(std + dataset_std - 2 * covmean))
            
    
    pose_dist = np.array(pose_dist)
    # print(pose_dist)
    return np.mean(pose_dist), np.std(pose_dist)


if __name__ == '__main__':
    args = {
        'dataset_directory': './dataset/amass/SAMPLED_POSES/',
        'generated_directory': 'experiments/samples/NRDF/',
        'dataset_data': 'experiments/utils/dataset_mean_cov.npz',
        'model': './dataset/models/neutral/model.npz',

        'faiss_model': 'dataset/amass/FAISS_MODEL',
        'k-faiss': 1000,
        'k-dist': 1,

        'dataset_name': 'pose',

        'dataset_size': 500,
        

    }

    # print(average_pairwise_distance(args))
    print(distance_gen_dataset(args))
    # print(frechet_distance(args))
