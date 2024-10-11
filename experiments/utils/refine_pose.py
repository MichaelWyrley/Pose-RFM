import sys
# add the current working directory so this can be run from the github repo root !!
sys.path.append('')
import torch
import numpy as np
import glob
from os import path as osp

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero

import pytorch3d.transforms as transforms

def pose_to_vert(pose_body, betas, args, device='cuda'):
    num_betas = len(betas) # number of body parameters
    time_length = len(pose_body)

    bm = BodyModel(args['model'], num_betas=num_betas, model_type='smplh').to(device)

    # Move body model so that it is in camera view !!
    trans = torch.Tensor([0,0,10]).to(device)
    trans = trans.unsqueeze(0).expand(time_length,-1)

    betas = betas.unsqueeze(0).expand(time_length,-1)
    pose_body = pose_body.reshape(time_length, -1)
    body_pose_beta = bm(pose_body=pose_body, betas=betas, trans=trans)
    return body_pose_beta


# modified from https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py#L60
def gmof(x, sigma):
    squared_x = x ** 2
    dist = torch.div(squared_x, squared_x + sigma ** 2)
    return sigma ** 2 * dist

# project 3d pose to 2d camera
def cam_3d_to_2d(joints, cam_intrinsics):
    pose_2d = torch.einsum('ik,BJk->BJi', cam_intrinsics, joints)

    # Output pixel coordinates
    return pose_2d[...,:2] / pose_2d[...,2].unsqueeze(-1)

# Calculate the Mahalanobis distance
def beta_dist(x, mu, sigma_inv):
    return torch.sqrt((x - mu).T @ sigma_inv @ (x - mu))


def calculate_prior_term(initial_pose, initial_betas, cam_intrinsics, model, args, device):

    joints = pose_to_vert(initial_pose, initial_betas, args, device).Jtr[:,:22]
    actual_pose_2d = cam_3d_to_2d(joints, cam_intrinsics)

    optimise_pose = initial_pose.clone().requires_grad_(True)
    optimise_betas = initial_betas.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([optimise_pose, optimise_betas], lr=args['learn_rate'], betas=(0.9, 0.999))
    # optimizer = torch.optim.LBFGS([optimise_pose, optimise_betas], lr=args['learn_rate'], max_iter=args['LBFGS_max_iter'])
    previous_loss = 9999999999999
    greater_values = 0
    for i in range(args['refine_max_steps']):
        optimizer.zero_grad()

        # Calculate data_loss from 2d projected joints
        optimise_joints = pose_to_vert(optimise_pose, optimise_betas, args, device).Jtr[:,:22]
        optimise_2d = cam_3d_to_2d(optimise_joints, cam_intrinsics)
        data_loss = (gmof(optimise_2d - actual_pose_2d, args['sigma']).sum(-1)).sum(-1)

        # Denoise pose
        predicted_pose = model.denoise_pose(optimise_pose, args['initial_timestep'], args['timesteps'], args['scale'])
        # Calculate pose loss as distance traveled for denoising
        optimise_mx = transforms.euler_angles_to_matrix(optimise_pose, "XYZ").reshape(-1,3,3)
        predicted_mx = transforms.euler_angles_to_matrix(predicted_pose, "XYZ").reshape(-1,3,3)
        pose_loss = transforms.so3_relative_angle(optimise_mx, predicted_mx).reshape(predicted_pose.shape[0], predicted_pose.shape[1]).mean(-1)

        # Calculate beta loss as the Mahalanobis distance between optimsed betas and average amass betas
        beta_loss = (beta_dist(optimise_betas, args['mean_betas'], args['betas_cov_inv']) ** 2).unsqueeze(0).expand(data_loss.shape[0])


        loss = (data_loss * args['data_loss_regulariser'] + pose_loss * args['pose_loss_regulariser'] + beta_loss * args['betas_loss_regulariser']).sum()
        loss.backward()
        optimizer.step()
        print(f"iteration {i}", loss.item(), (data_loss * args['data_loss_regulariser']).sum().item(), ( args['pose_loss_regulariser'] *pose_loss).sum().item(), (beta_loss * args['betas_loss_regulariser']).sum().item())

        if loss.item() > previous_loss:
            greater_values += 1

        elif greater_values > args['amount_of_greater_values']: 
            break
        
        previous_loss = loss.item()

    return optimise_pose, optimise_betas


def refing_poses(args, device='cuda'):
    model = DiT_adaLN_zero(in_dim=6, depth=12, emb_dimention=768, num_heads=12,).to(device)
    model.load_state_dict(torch.load(args['load_model']))
    model.eval()

    model = FlowMatchingMatrix(model, device=device)

    poses_hat_seq = sorted(glob.glob(args['generated_directory'] + '/*.npz'))
    poses_gt = sorted(glob.glob(args['ground_truth_directory'] + '/*.npz'))

    betas_data = np.load(args['average_betas_information'])
    args['mean_betas'] = torch.Tensor(betas_data['mean_betas']).to(device)
    args['betas_cov_inv'] = torch.Tensor(np.linalg.inv(betas_data['betas_cov'])).to(device)

    for i, (seq, gt_sequence) in enumerate(zip(poses_hat_seq, poses_gt)):
        name = seq.split('/')[-1].split('.')[0]
        print("refining " + name)
        cdata = np.load(seq, allow_pickle=True)
        gt_data = np.load(gt_sequence, allow_pickle=True)
        poses_hat = torch.Tensor((cdata['body_pose'][:, :63]).reshape(-1, 21, 3)).to(device)
        betas_hat = torch.Tensor(cdata['betas']).to(device)
        cam_intrinsics = torch.Tensor(gt_data['cam_intrinsics']).to(device)

        batched_poses_hat = torch.split(poses_hat, args['batch_size'])
        refined_poses = []
        refined_betas = np.zeros_like(cdata['betas'], dtype=np.float64)
        for pose_hat in batched_poses_hat:
            pose, betas = calculate_prior_term(pose_hat, betas_hat, cam_intrinsics, model, args, device)

            refined_poses.append(pose.cpu().detach().numpy())
            refined_betas += betas.cpu().detach().numpy()

        refined_poses = np.concatenate(refined_poses, axis=0).reshape(poses_hat.shape[0], -1)
        refined_betas = refined_betas / len(batched_poses_hat)

        np.savez(args['denoised_directory'] + name, body_pose=refined_poses, betas=refined_betas)


if __name__ == '__main__':
    args = {

        'load_model': 'best_model/ema_model_1200.pt',

        'ground_truth_directory': 'dataset/3DPW/npz_poses/ground_truth/',
        'generated_directory': 'dataset/3DPW/smpl_poses/',
        'denoised_directory': 'dataset/3DPW/npz_poses/nrdf_smpl/',

        'batch_size': 500,
        'initial_timestep': 10,
        'timesteps': 15,
        'scale': 1,
        
        'data_loss_regulariser': 0.0001,
        'pose_loss_regulariser': 3.5,
        'betas_loss_regulariser': 0.15,
        'general_loss_regulariser': 0,
        'refine_max_steps': 25,
        'amount_of_greater_values': 3,
        # 'LBFGS_max_iter': 30,
        'learn_rate': 0.1,
        'sigma': 10,

        'average_betas_information': 'experiments/utils/betas.npz',
        
        'model': 'dataset/models/neutral/model.npz',
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    refing_poses(args, device)
    