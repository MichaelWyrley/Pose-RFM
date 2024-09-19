# Modified from https://github.com/hynann/NRDF/blob/master/lib/exp/pose_den.py#L187
import sys
sys.path.append('')

import os
import os.path as osp
import argparse
import numpy as np
import torch
import datetime

from FlowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from VectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero
from Utils.NRDF.data.gen_data import nn_search
from Utils.NRDF.utils.data_utils import geo, load_faiss,quat_to_global
from Utils.NRDF.utils.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

def pose_to_vert(pose_body, bm_path, device='cuda'):
    num_betas = 16 # number of body parameters

    bm = BodyModel(bm_path, num_betas=num_betas, model_type='smplh').to(device)
    time_length = len(pose_body)
  
    body_pose_beta = bm(pose_body=torch.Tensor(pose_body.reshape(time_length, -1)).to(device))
    return body_pose_beta.Jtr

def cal_dist_geo(nonman_pose, man_pose):
    dis = torch.mean(1 - torch.abs(torch.sum(man_pose*nonman_pose, dim=2)), dim=1)
    return dis

def v2v_err(pose_src, pose_dst, bm_path, device='cuda'):
    
    verts_src, verts_dst = pose_to_vert(pose_src,bm_path, device)[:,:22], pose_to_vert(pose_dst,bm_path, device)[:,:22]
    
    m2m_dist = verts_src - verts_dst
    m2m_dist = torch.mean(torch.sqrt(torch.sum(m2m_dist * m2m_dist, dim=-1)), dim=-1).detach().cpu().numpy()  # (N, )
    
    return m2m_dist

def quat_geo_global(pose_src, pose_dst, device='cuda'):
    quat_src = axis_angle_to_quaternion(torch.Tensor(pose_src).to(device))
    quat_dst = axis_angle_to_quaternion(torch.Tensor(pose_dst).to(device))

    quat_src_glob, quat_dst_glob = quat_to_global(quat_src), quat_to_global(quat_dst)
    geo_dist_glob = torch.mean(cal_dist_geo(quat_src_glob, quat_dst_glob), dim=-1).detach().cpu().numpy()
    
    return geo_dist_glob

class Projector(object):
    def __init__(self, model, noisy_pose_path=None, noisy_pose=None, device='cuda:0', batch_size=100):
        """

        Args:
            model: flow matching model
            noisy_pose_path: [optional] Path to the input noisy pose file, npz format
            noisy_pose: [optional] Input noisy pose: numpy [bs, nj*3]
            device: cuda or cpu

        """
        self.device = device

        self.batch_size = batch_size
        

        # load pretrained NRDF model
        self.model = model

        # initialize input noisy pose
        self.noisy_pose = None
        if noisy_pose_path is not None:
            self.noisy_pose = self._load_noisy_pose(noisy_pose_path)  # torch [bs, nj*3]
        else:
            # self.noisy_pose = torch.from_numpy(noisy_pose).to(torch.float32).to(self.device) # torch [bs, nj*3]
            self.noisy_pose = noisy_pose # torch [bs, nj*3]
            self.noisy_pose = self.noisy_pose.to(torch.float32).to(self.device)

    def project(self, initial_timestep, timestep, scale):
        """

        Args:
            step_size: alpha in Eq.(8)
            iterations: Max iteration during projection
            save_all_step: if true: save intermediate poses in all projection steps, else save the converged poses only

        Returns: step_aa: numpy [bs, nj, 3], result poses in all projection steps, axis-angle format

        """
        
        clean_poses = self.model.denoise_pose(self.noisy_pose, initial_timestep, timestep, scale)

        return clean_poses

    def _load_noisy_pose(self, path, subsample=True):
        noisy_pose = np.load(path)['noisy_pose_aa']
        

        subsample_indices = np.random.randint(0, len(noisy_pose), self.batch_size)
        noisy_pose = noisy_pose[subsample_indices]
        
        noisy_pose = torch.from_numpy(noisy_pose).to(torch.float32).to(self.device)
        noisy_pose = noisy_pose.reshape(-1, 21, 3)

        return noisy_pose
       
    
    def cal_error(self, converged_pose,
                        bm_path,
                        faiss_model_dir,
                        k_faiss=1000,
                        k_dist=1):
        """

        Args:
            pose_aa: numpy [bs, nj, 3], result pose in all projection steps, axis-angle format
            bm_path: SMPL body model path
            faiss_model_dir: Pretrained faiss model path
            k_faiss: Number of candidates selected by the kd tree
            k_dist: Number of final nearest neighbor

        """

        # load faiss related terms
        index, all_poses_aa, all_poses_quat = load_faiss(faiss_model_dir)

        pose_quat = axis_angle_to_quaternion(converged_pose).detach().cpu().numpy() # [bs, nj, 4]
        dist_cal = geo()

        # search nearest neighbors
        k_quats, k_poses_aa, dist_gt = nn_search(quat=pose_quat, 
                                              index=index, 
                                              dist_cal=dist_cal, 
                                              all_poses_aa=all_poses_aa, 
                                              all_poses_quat=all_poses_quat, 
                                              k_faiss=k_faiss, 
                                              k_dist=k_dist)
        
        nn_pose = k_poses_aa[:, 0].reshape(-1,21,3) # nearest neighbor [bs, 63]

        # caculate v2v
        m2m_dist = v2v_err(converged_pose, nn_pose, bm_path=bm_path, device=self.device)
        geo_dist_glob = quat_geo_global(converged_pose, nn_pose, device=self.device)

        geo_m2m = 0.5 * geo_dist_glob + m2m_dist
        geo_m2m, m2m_dist = geo_m2m.mean(), m2m_dist.mean()

        print(f'delta_q+m2m error: {geo_m2m}')
        print(f'marker2marker error: {m2m_dist}')

        return geo_m2m, m2m_dist


def project_poses(model, args):
    projector = Projector(model=model,
                          noisy_pose_path=args['directory'] + args['noisy'],
                          device=args['device'], batch_size = args['batch_size'])

    geo_m2ms, m2m_dists = [], []

    for i in range(args['no_samples']):
        res_aa = projector.project(initial_timestep = args['initial_timestep'], timestep = args['timesteps'], scale = args['scale'])
        
        print('Caculating metrics...')
        geo_m2m, m2m_dist = projector.cal_error(res_aa,
                            bm_path=args['directory'] + args['model'],
                            faiss_model_dir=args['directory'] + args['faiss_model'],
                            k_faiss=args['k-faiss'],
                            k_dist=args['k-dist'])

        geo_m2ms.append(geo_m2m)
        m2m_dists.append(m2m_dist)
    
    geo_m2ms = np.array(geo_m2ms)
    m2m_dists = np.array(m2m_dists)
    print(geo_m2ms.mean(), geo_m2ms.std(), m2m_dists.mean(), m2m_dists.std())
    return geo_m2ms.mean(), geo_m2ms.std(), m2m_dists.mean(), m2m_dists.std()


if __name__ == "__main__":

    args = {
        # 'support_dir': '/vol/bitbucket/mew23/individual_project/',
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'noisy': 'dataset/amass/NOISY_POSES/examples/noisy_pose.npz',

        'save_location': 'experiments/samples/denoised_pose/',
        'model': './dataset/models/neutral/model.npz',

        'faiss_model': 'dataset/amass/FAISS_MODEL',
        'k-faiss': 1000,
        'k-dist': 1,

        'load_model': 'working_models/noised_pose_90/ema_model_1200.pt',
        'dataset_name': 'pose_body',

        'batch_size': 500,
        'initial_timestep': 5,
        'timesteps': 15,
        'scale': 4,

        'no_samples': 1,

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    model = DiT_adaLN_zero(in_dim=6, depth=12, emb_dimention=768, num_heads=12,).to(args['device'])
    model.load_state_dict(torch.load(args['directory']+args['load_model']))
    model.eval()

    model = FlowMatchingMatrix(model, device=args['device'])
    
    project_poses(model, args)
