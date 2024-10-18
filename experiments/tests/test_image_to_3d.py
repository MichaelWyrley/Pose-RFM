
import sys
# add the current working directory so this can be run from the github repo root !!
sys.path.append('')
import torch
import numpy as np
import glob
from os import path as osp

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

# from experiments.utils.refine_pose import refing_poses


def pose_to_vert(pose_body, betas, root_orient, trans, args, num_betas = 10, device='cuda'):
    time_length = len(pose_body)

    bm = BodyModel(args['model'], num_betas=num_betas, model_type='smplh').to(device)
    if len(betas.shape) == 1: 
        betas = torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device)
    else:
        betas = torch.Tensor(betas).to(device)


    pose_body = torch.Tensor(pose_body.reshape(time_length, -1)).to(device)
    root_orient = torch.Tensor(root_orient).to(device)
    # trans = torch.Tensor(np.repeat(trans[:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device)
    trans = torch.Tensor(trans).to(device)

    body_pose_beta = bm(pose_body=pose_body, betas=betas, root_orient = root_orient, trans = trans)
    return body_pose_beta


# modified from https://github.com/caizhongang/SMPLer-X/blob/main/common/utils/transforms.py#L33
def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

# modified from https://github.com/caizhongang/SMPLer-X/blob/main/common/utils/transforms.py#L52
def rigid_align(A, B):
    # A1, B1 = A.reshape(A.shape[0], -1), B.reshape(B.shape[0],-1)
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2

# modified from https://github.com/caizhongang/SMPLer-X/blob/main/data/humandata.py#L643
def get_mpjpe(gtf_poses, genf_poses):
    joint_out_body_align = rigid_align(genf_poses, gtf_poses)
    dist =  np.sqrt(np.sum((joint_out_body_align - gtf_poses) ** 2, 1)) * 1000

    return dist.mean()

# modified from https://github.com/caizhongang/SMPLer-X/blob/main/data/humandata.py#593
def get_pve(gtf_vert, genf_verts):
    # MPVPE from all vertices
    mesh_out_align = rigid_align(genf_verts, gtf_vert)
    pa_mpvpe = np.sqrt(np.sum((mesh_out_align - gtf_vert) ** 2, 1)) * 1000


    return pa_mpvpe.mean()

def get_pck(gtf_poses, genf_poses):
    joint_out_body_align = rigid_align(genf_poses, gtf_poses)
    distance = np.sqrt(np.sum((joint_out_body_align - gtf_poses) ** 2, 1)) * 1000
    
    percent = (distance < 100).mean()

    return percent

def calculate_metrics(ground_truth_dir, generated_dir, args, device='cuda'):
    
    ground_truth_files = sorted(glob.glob(ground_truth_dir + '*.npz'))[1:]
    generated_files = sorted(glob.glob(generated_dir + '*.npz'))[1:]

    results = {'pa_mpjpe': [], 'pa_mpvpe': [], 'pck': [] }

    for (gtf, genf) in zip(ground_truth_files, generated_files):
        # print("Calculateing for:", gtf.split('/')[-1].split('.')[0], genf.split('/')[-1].split('.')[0])
        result = {'pa_mpjpe': [], 'pa_mpvpe': [], 'pck': [] }

        ground_truth_sequence = np.load(gtf, allow_pickle=True)
        generated_sequence = np.load(genf, allow_pickle=True)

        gtf_poses = np.array(ground_truth_sequence['poses'][0])[:, 3:66]  # 3:72 then :63
        gtf_root_orient = np.array(ground_truth_sequence['poses'][0])[:, :3]
        gtf_trans = np.array(ground_truth_sequence['trans'][0, :, :])
        gtf_betas = np.array(ground_truth_sequence['betas'][0])

        genf_poses = np.array(generated_sequence['body_pose'])[:, :63]
        genf_root_orient = np.array(generated_sequence['global_orient'])[:, :3]
        if not len(generated_sequence['trans'].shape) == 2:
            genf_trans = np.array(generated_sequence['trans'][:, 0, :])
        else:
            genf_trans = np.array(generated_sequence['trans'])
        genf_betas = np.array(generated_sequence['betas'])

        gtf_poses = pose_to_vert(gtf_poses, gtf_betas, gtf_root_orient, gtf_trans, args)
        genf_poses = pose_to_vert(genf_poses, genf_betas, genf_root_orient, genf_trans, args)
            

        for i in range(genf_poses.Jtr.shape[0]):
            result['pa_mpjpe'].append( get_mpjpe(gtf_poses.Jtr[i].cpu().detach().numpy(), genf_poses.Jtr[i].cpu().detach().numpy()))
            result['pa_mpvpe'].append( get_pve(gtf_poses.v[i].cpu().detach().numpy(), genf_poses.v[i].cpu().detach().numpy()))
            result['pck'].append(      get_pck(gtf_poses.Jtr[i].cpu().detach().numpy(), genf_poses.Jtr[i].cpu().detach().numpy()))


        results['pa_mpjpe'].append(np.array( result['pa_mpjpe']).mean())
        results['pa_mpvpe'].append(np.array( result['pa_mpvpe']).mean())
        results['pck'].append(np.array( result['pck']).mean())


    results['pa_mpjpe'] = np.array( results['pa_mpjpe'])
    results['pa_mpvpe'] = np.array( results['pa_mpvpe'])
    results['pck'] = np.array( results['pck'])
    
    print('pa_mpjpe:', results['pa_mpjpe'].mean(),  results['pa_mpjpe'])
    print('pa_mpvpe:', results['pa_mpvpe'].mean(),  results['pa_mpvpe'])
    print('pck:',      results['pck'].mean(),       results['pck'])

    return results


if __name__ == '__main__':
    args = {

        'load_model': 'best_model/ema_model_1200.pt',

        'ground_truth_directory': 'dataset/3DPW/npz_poses/ground_truth/',
        'generated_directory': '../DPoser/dpose_results_no_prior/',
        # 'denoised_directory': 'dataset/3DPW/npz_poses/dpose_results_no_prior/',
        'all_dirs': ['dataset/3DPW/dpose_poses_100/dpose_results_no_prior/', 'dataset/3DPW/dpose_poses_100/dpose_results_vpose/', 'dataset/3DPW/dpose_poses_100/dpose_results_pose_ndf/', 'dataset/3DPW/dpose_poses_100/dpose_results_nrdf/','dataset/3DPW/dpose_poses_100/dpose_results_pose_rfm/'],
        
        'model': 'dataset/models/neutral/model.npz',
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # refing_poses(args, device)
    for i in args['all_dirs']:
        print("calculating for:", i.split('/')[-2])
        results = calculate_metrics(args['ground_truth_directory'], i, args, device=device)
    
    # result_gt = calculate_metrics(args['ground_truth_directory'], args['generated_directory'], args, device=device)
    # result_denoised = calculate_metrics(args['ground_truth_directory'], args['denoised_directory'], args, device=device)

    