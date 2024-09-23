import sys
sys.path.append('')
import torch
import os
from os import path as osp

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero
from main.utils.NRDF.utils.data_utils import amass_splits

from human_body_prior.body_model.body_model import BodyModel
import numpy as np

from scipy.linalg import sqrtm

from main.utils.NRDF.data.gen_data import nn_search
from main.utils.NRDF.utils.data_utils import geo, load_faiss
from main.utils.NRDF.utils.transforms import axis_angle_to_quaternion

import glob

# Clean up so this is only in one place !!
def pose_to_vert(pose_body, args, device='cuda'):
    num_betas = 16 # number of body parameters

    bm = BodyModel(args['model'], num_betas=num_betas, model_type='smplh').to(pose_body.device)
    time_length = len(pose_body)
  
    body_pose_beta = bm(pose_body=pose_body.reshape(time_length, -1))
    return body_pose_beta.Jtr

def gen_masks(args):
    sample_removal = torch.rand((1, 21))
    sample_removal = sample_removal > args['removal_level']

    return sample_removal

def calculate_pairwise_distance(gen_pose, no_masks):
    num_poses = gen_pose.shape[0] // no_masks

    total_distance = np.zeros(num_poses)

    for k in range(num_poses):
        total_sum = 0
        for i in range(no_masks):
            for j in range(i+1, no_masks):
                total_distance[k] += np.mean(np.linalg.norm(gen_pose[num_poses * k + i] - gen_pose[num_poses * k + j], axis = 1))
                total_sum += 1
        total_distance[k] = total_distance[k] / total_sum

    return total_distance.mean()

def fid_calculateion(gen_pose, dataset_mu, dataset_std):
    gen_mean, gen_cov = np.mean(gen_pose, axis=0), np.cov(gen_pose, rowvar=False)

    ssdiff = np.sum((gen_mean - dataset_mu)**2)
    covmean, _ = sqrtm(np.dot(gen_cov, dataset_std), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(gen_cov + dataset_std - 2 * covmean)

    return dist

def dnn(gen_pose, index, all_poses_aa, all_poses_quat, k_faiss=1000, k_dist=1):
    pose_dist = []
        
    pose_body_quat = axis_angle_to_quaternion(gen_pose).detach().cpu().numpy() # [bs, nj, 4]
    dist_cal = geo()

    # search nearest neighbors
    _, _, dist_gt = nn_search(quat=pose_body_quat, 
                                            index=index, 
                                            dist_cal=dist_cal, 
                                            all_poses_aa=all_poses_aa, 
                                            all_poses_quat=all_poses_quat, 
                                            k_faiss=k_faiss, 
                                            k_dist=k_dist)

    
    return dist_gt.mean()

# Actually randomly chose the mask, and generate a series of masked poses based on a single pose with a single mask !!!! 
# Do this for multiple randomly chosen masks 
# And test that !!
def gen_partial(diffusion, args):
    device = diffusion.device
    clean_seqs = sorted(glob.glob(args['clean'] + '/*/*.npz'))
    clean_seqs = [ds for ds in clean_seqs if ds.split('/')[-2] in amass_splits['test']]

    if args['gen_masks']:
        masks = gen_masks(args).to(device)
        masks = masks.expand(args['batch_size'],-1)

        np.savez(args['save_location'] + 'mask.npz', pose_body=masks.cpu().numpy())
    else: 
        np.load(args['save_location'] + 'mask.npz')

    total_APD = []
    total_FID = []
    total_dnn = []

    cdata = np.load(args['dataset_data'])
    dataset_mu, dataset_std = cdata['mu'], cdata['std']

    
    # load faiss related terms
    index, all_poses_aa, all_poses_quat = load_faiss(args['faiss_model'])

    with torch.no_grad():
        for k in range(args['no_samples']):
            print("Sample: ",k)
            
            apd_dist = []
            fid_dist = []
            dnn_dist = []
            for i, seq in enumerate(clean_seqs):
                print("starting " + seq.split('/')[-1].split('.')[0])
                cdata = np.load(seq)
                clean_poses = cdata['pose_body'][:, :63].astype(np.float32)
                clean_poses = torch.Tensor(clean_poses.reshape(-1, 21, 3)).to(device)

                subsample_indices = np.random.randint(0, len(clean_poses), args['random_subset'])
                clean_poses = clean_poses[subsample_indices]

                batched_clean_poses = torch.split(clean_poses, args['batch_size'] // args['no_masks'])
                
                for clean_pose in batched_clean_poses:
                    poses = torch.repeat_interleave(clean_pose, args['no_masks'], dim=0)
                    
                    gen_poses = diffusion.sample_partial(poses, masks[:poses.shape[0]], args['timesteps'], args['scale'], args['stop_sampling'])

                    dnn_dist.append(dnn(gen_poses, index, all_poses_aa, all_poses_quat))

                    gen_poses_vert = pose_to_vert(gen_poses, args)[:, :22].cpu().numpy()
                    apd_dist.append(calculate_pairwise_distance(gen_poses_vert, args['no_masks']))
                    fid_dist.append(fid_calculateion(gen_poses_vert.reshape(-1, 66), dataset_mu, dataset_std))

            dnn_dist = np.array(dnn_dist)
            apd_dist = np.array(apd_dist)
            fid_dist = np.array(fid_dist)
            total_FID.append(np.mean(fid_dist))
            total_APD.append(np.mean(apd_dist))
            total_dnn.append(np.mean(dnn_dist))
    
    total_APD = np.array(total_APD)
    total_FID = np.array(total_FID)
    total_dnn = np.array(total_dnn)
    return np.mean(total_APD), np.std(total_APD), np.mean(total_dnn), np.std(total_dnn),np.mean(total_FID), np.std(total_FID)
            
            
    

if __name__ == '__main__':

    args = {
        'clean': './dataset/amass/SAMPLED_POSES/',
        'model': './dataset/models/neutral/model.npz',

        'load_model': 'working_models/noised_pose_90/model_1200.pt',

        'save_location': 'experiments/samples/partial_generation/', 
        'scale': 3.5,
        'timesteps': 10,
        'removal_level': 0.2,
        'no_masks': 50, 
        'batch_size': 500,
        'gen_masks': True,
        'dataset_data': 'experiments/utils/dataset_mean_cov.npz',
        'random_subset': 50,

        'stop_sampling': 1,

        'no_samples': 20,

        'faiss_model': 'dataset/amass/FAISS_MODEL',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero(in_dim=6, depth=12, emb_dimention=768, num_heads=12,).to(device)
    model.load_state_dict(torch.load(args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    print(gen_partial(diffusion, args))


