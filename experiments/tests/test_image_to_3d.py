
import sys
# add the current working directory so this can be run from the github repo root !!
sys.path.append('')
import torch
import numpy as np
import glob
from os import path as osp

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

from experiments.utils.refine_pose import refing_poses


def pose_to_vert(pose_body, betas, args, device='cuda'):
    num_betas = len(betas) # number of body parameters
    time_length = len(pose_body)

    bm = BodyModel(args['model'], num_betas=num_betas, model_type='smplh').to(device)

    betas = torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device)
    pose_body = torch.Tensor(pose_body.reshape(time_length, -1)).to(device)
    body_pose_beta = bm(pose_body=pose_body, betas=betas)
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

    return dist

# modified from https://github.com/caizhongang/SMPLer-X/blob/main/data/humandata.py#593
def get_pve(gtf_vert, genf_verts):
    # MPVPE from all vertices
    mesh_out_align = rigid_align(genf_verts, gtf_vert)
    pa_mpvpe = np.sqrt(np.sum((mesh_out_align - gtf_vert) ** 2, 1)) * 1000


    return pa_mpvpe.mean()

def get_pck(gtf_poses, genf_poses):
    joint_out_body_align = rigid_align(genf_poses, gtf_poses)
    distance = np.sqrt(np.sum((joint_out_body_align - gtf_poses) ** 2, 1)) * 1000
    
    percent = (distance < 50).mean()

    return percent

def calculate_metrics(ground_truth_dir, generated_dir, args, ground_truth = False, device='cuda'):
    
    ground_truth_files = sorted(glob.glob(ground_truth_dir + '*.npz'))
    generated_files = sorted(glob.glob(generated_dir + '*.npz'))

    results = {'pa_mpjpe': [], 'pa_mpvpe': [], 'pck': [] }

    for (gtf, genf) in zip(ground_truth_files, generated_files):
        print("Calculateing for:", gtf.split('/')[-1].split('.')[0], genf.split('/')[-1].split('.')[0])
        result = {'pa_mpjpe': [], 'pa_mpvpe': [], 'pck': [] }

        ground_truth_sequence = np.load(gtf, allow_pickle=True)
        generated_sequence = np.load(genf, allow_pickle=True)

        gtf_poses = np.array(ground_truth_sequence['poses'][0])[:, 3:66]  # 3:72 then :63
        gtf_betas = np.array(ground_truth_sequence['betas'][0])[:10]

        genf_poses = np.array(generated_sequence['body_pose'])[:, :63]
        genf_betas = np.array(generated_sequence['betas'])[:10]

        gtf_poses = pose_to_vert(gtf_poses, gtf_betas, args)
        genf_poses = pose_to_vert(genf_poses, genf_betas, args)
            

        for i in range(gtf_poses.Jtr.shape[0]):
            result['pa_mpjpe'].append( get_mpjpe(gtf_poses.Jtr[i].cpu().detach().numpy()[:22], genf_poses.Jtr[i].cpu().detach().numpy()[:22]))
            result['pa_mpvpe'].append( get_pve(gtf_poses.v[i].cpu().detach().numpy(), genf_poses.v[i].cpu().detach().numpy()))
            result['pck'].append(      get_pck(gtf_poses.Jtr[i].cpu().detach().numpy()[:22], genf_poses.Jtr[i].cpu().detach().numpy()[:22]))

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
        'generated_directory': 'dataset/3DPW/smpl_poses/',
        'denoised_directory': 'dataset/3DPW/npz_poses/nrdf_smpl/',

        'batch_size': 500,
        'initial_timestep': 23,
        'timesteps': 25,
        'scale': 1,
        
        'data_loss_regulariser': 0.02,
        'pose_loss_regulariser': 0.5,
        'betas_loss_regulariser': 0.005,
        'general_loss_regulariser': 0,
        'refine_max_steps': 25,
        'amount_of_greater_values': 3,
        # 'LBFGS_max_iter': 30,
        'learn_rate': 0.01,
        'sigma': 10,

        'average_betas_information': 'experiments/utils/betas.npz',
        
        'model': 'dataset/models/neutral/model.npz',
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    refing_poses(args, device)
    result_gt = calculate_metrics(args['ground_truth_directory'], args['generated_directory'], args, ground_truth = True, device=device)
    result_denoised = calculate_metrics(args['ground_truth_directory'], args['denoised_directory'], args, ground_truth=False,  device=device)

    