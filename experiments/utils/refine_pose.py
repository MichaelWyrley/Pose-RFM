import sys
# add the current working directory so this can be run from the github repo root !!
sys.path.append('')
import torch
import numpy as np
import glob
from os import path as osp

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel



import pytorch3d.transforms as transforms
from experiments.utils.DPose.simplify import SMPLify
from experiments.utils.DPose.joint_mapping import mmpose_to_openpose, vitpose_to_openpose
from experiments.utils.DPose.fitting_losses import guess_init
from experiments.utils.DPose.constants import BEND_POSE_PATH


from experiments.utils.DPose.smpl import SMPL


# # Calculate the Mahalanobis distance
# def beta_dist(x, mu, sigma_inv):
#     return torch.sqrt((x - mu).T @ sigma_inv @ (x - mu))

# Partially Modified from https://github.com/moonbow721/DPoser/blob/v2/run/tester/body/EHF.py
def calculate_prior_term(initial_pose, initial_betas, coords_2d, bbox, trans, cam_intrinsics, args, device, frame, name):

    mm_keypoints = coords_2d[:, :, :2]
    keypoint_scores = coords_2d[:, :, 2]
    keypoints = mmpose_to_openpose(mm_keypoints, keypoint_scores)[:, :25]

    bbox_centre = (bbox[:, :2] + bbox[:, 2:4]) / 2

    # Transform the keypoints to the camera space
    camera_centre = cam_intrinsics[:2, 2]
    diff = bbox_centre - camera_centre.unsqueeze(0)
    keypoints_transformed = keypoints[:,:,:2] - diff.unsqueeze(1)
    keypoints_transformed = torch.cat([keypoints_transformed, keypoints[:,:,2].unsqueeze(-1)], dim=-1)
    
    # # be careful: the estimated focal_length should be used here instead of the default constant
    smplify = SMPLify(cam_intrinsics=cam_intrinsics, step_size=args['learning_rate'], batch_size=initial_pose.shape[0], num_iters_cam=args['num_iters_cam'], num_iters_pose=args['num_iters_pose'],
                        device = device, args=args)
    

    # bend_init = bbox[:, 2] > 400
    # initial_pose = smplify.body_model.mean_poses.unsqueeze(0).repeat(coords_2d.shape[0], 1).to(device)  # N*66
    # bend_pose = torch.from_numpy(np.load(BEND_POSE_PATH)['pose']).to(device)
    # initial_pose[bend_init, 3:] = bend_pose[:, 3:]
    # initial_betas = smplify.body_model.mean_shape.unsqueeze(0).repeat(coords_2d.shape[0], 1)  # N*10

    if len(initial_betas.shape) == 1:
        initial_betas = initial_betas.unsqueeze(0).repeat(coords_2d.shape[0], 1)
    
    smpl_output = smplify.pose_to_vert(betas=initial_betas,
                                        pose_body=initial_pose[:, 3:],
                                        global_orient=initial_pose[:, :3],
                                        trans=trans / 10.,
                                        args=args)
    guess_init_t = guess_init(smpl_output.joints, keypoints_transformed, cam_intrinsics, device)
    
    
    results = smplify(initial_pose.detach(),
                        initial_betas.detach(),
                        guess_init_t.detach(),
                        camera_centre,
                        keypoints_transformed)

    new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results

    # smplify.output_image(initial_pose, initial_betas, guess_init_t , keypoints_transformed, diff, args['imag_loc'] + name, initial_pose.shape[0] * frame, args['save_loc'] + name)
    # exit()

    return new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss


def refing_poses(args, device='cuda'):

    poses_hat_seq = sorted(glob.glob(args['generated_directory'] + '/*.npz'))
    poses_gt_seq = sorted(glob.glob(args['ground_truth_directory'] + '/*.npz'))
    coords_2d_seq = sorted(glob.glob(args['2d_coords'] + '/*.npz'))

    betas_data = np.load(args['average_betas_information'])
    args['mean_betas'] = torch.Tensor(betas_data['mean_betas']).to(device)
    args['betas_cov_inv'] = torch.Tensor(np.linalg.inv(betas_data['betas_cov'])).to(device)

    for i, (seq, gt_seq, coords_2d_seq) in enumerate(zip(poses_hat_seq, poses_gt_seq, coords_2d_seq)):
        name = seq.split('/')[-1].split('.')[0]
        print("refining " + name)
        pose_data = np.load(seq, allow_pickle=True)
        gt_data = np.load(gt_seq, allow_pickle=True)
        coords_2d_data = np.load(coords_2d_seq, allow_pickle=True)

        poses_hat = torch.Tensor((pose_data['full_pose'])).to(device)
        trans = torch.Tensor(pose_data['trans'][:, 0, :]).to(device)
        betas_hat = torch.Tensor(pose_data['betas']).to(device)
        cam_intrinsics = torch.Tensor(gt_data['cam_intrinsics']).to(device)

        bbox_2d_data = torch.Tensor(coords_2d_data['bbox']).to(device)
        coords_2d_data = torch.Tensor(coords_2d_data['keypoints']).to(device)

        batched_poses_hat = torch.split(poses_hat, args['batch_size'])
        batched_2d_coords = torch.split(coords_2d_data, args['batch_size'])
        batched_bbox_2d = torch.split(bbox_2d_data, args['batch_size'])
        batched_trans = torch.split(trans, args['batch_size'])
        refined_poses = []
        refined_cam_t = []
        refined_betas = []

        for i, (pose_hat, coords_2d, bbox, trans) in enumerate(zip(batched_poses_hat, batched_2d_coords, batched_bbox_2d, batched_trans)):
            new_pose, new_betas, new_opt_cam_t, new_opt_joint_loss = calculate_prior_term(pose_hat, betas_hat, coords_2d, bbox, trans, cam_intrinsics, args, device, i, name)
            
            refined_cam_t.append(new_opt_cam_t.cpu().detach().numpy())
            refined_poses.append(new_pose.cpu().detach().numpy())
            refined_betas.append(new_betas.cpu().detach().numpy())

        refined_poses = np.concatenate(refined_poses, axis=0).reshape(poses_hat.shape[0], -1)
        refined_cam_t = np.concatenate(refined_cam_t, axis=0)
        refined_betas = np.concatenate(refined_betas, axis=0)

        np.savez(args['denoised_directory'] + name, body_pose=refined_poses[:,3:], betas=refined_betas, trans=refined_cam_t, global_orient=refined_poses[:, :3])



if __name__ == '__main__':
    args = {

        'load_model': 'best_model/ema_model_1200.pt',

        'ground_truth_directory': 'dataset/3DPW/npz_poses/ground_truth',
        'generated_directory': 'dataset/3DPW/smpl_poses',
        'denoised_directory': 'dataset/3DPW/npz_poses/denoised_smpl_no_prior/',
        '2d_coords': 'dataset/3DPW/vit_poses',
        'average_betas_information': 'experiments/utils/betas.npz',

        'batch_size': 500,
        'initial_timestep': 10,
        'timesteps': 15,
        'scale': 1,

        'learning_rate': 1e-2,
        'num_iters_cam': 100,
        'num_iters_pose': 100,
        
        #random:1, fix:2, truncated annealing:3, t=0:4
        'time_strategies': 4,
        # Pose-RFM, DPose, ???, None
        'pose_prior': None,

        'imag_loc': 'dataset/3DPW/imageFiles/test/',
        'save_loc': 'samples/fitting/',

        
        'model': 'dataset/models/neutral/model.pkl',
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    refing_poses(args, device)
    