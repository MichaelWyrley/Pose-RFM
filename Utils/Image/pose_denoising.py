import sys
sys.path.append('')
import torch
import os
import matplotlib.pyplot as plt

from FlowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from VectorFieldModels.Transformer import DiT
from VectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero

from Utils.NRDF.utils.transforms import quaternion_to_axis_angle
from Utils.NRDF.data.dataloaders import PoseData
import pytorch3d.transforms as transforms
from visualise import visualise, vis_body_pose_beta
import numpy as np

def noise_pose(clean_pose, noise_level=0.1):
    noise = torch.randn_like(clean_pose)
    noisy_pose = clean_pose + noise * noise_level

    return noisy_pose


def denoise(diffusion, args):

    data_set = PoseData(mode='train', clean_dir=args['directory'] + args['clean'], batch_size=args['batch_size'], num_workers=6, num_pts=args['no_samples'], stage=1, flip=False, random_poses=False)
    data_loader = data_set.get_loader()

    # Sample the full path for x ('samples') data points 

    with torch.no_grad():
        for i, poses in enumerate(data_loader):
            clean_pose = poses.to(device).reshape(-1, 21, 3)
            noisy_pose = noise_pose(clean_pose, noise_level=args['noise_level'])

            sampled = diffusion.denoise_pose(noisy_pose, args['initial_timestep'], args['timesteps'], scale=args['scale'])

            denoised_pose = sampled.cpu().numpy()
            clean_pose = clean_pose.cpu().numpy()
            noisy_pose = noisy_pose.cpu().numpy()
            np.savez(args['directory'] + f'./samples/denoised/denoised_pose/data_{i}.npz', pose_body=denoised_pose)
            np.savez(args['directory'] + f'./samples/denoised/clean_pose/data_{i}.npz', pose_body=clean_pose)
            np.savez(args['directory'] + f'./samples/denoised/initially_noisy/data_{i}.npz', pose_body=noisy_pose)
            break
    

if __name__ == '__main__':
    args = {
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'clean': './dataset/amass/SAMPLED_POSES/',

        'batch_size': 1,
        'no_samples': 16,
        'scale': 2,
        'noise_level': 0.2,

        'initial_timestep': 90,
        'timesteps': 100,
        'load_model': 'models/ema_model_400.pt',
        'sample_dir': 'samples/sample_list/',

        'frame': 'samples/denoised/clean_pose/data_0.npz',
        'image_loc': 'samples/denoised/clean_pose/images/',
        'model': './dataset/models/neutral/model.npz',

        'name': '',
        'print': False,
        'save_grid': True,

    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero().to(device)
    model.load_state_dict(torch.load(args['directory'] + args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)


    denoise(diffusion, args)
    visualise(args)
    args['frame'] = 'samples/denoised/denoised_pose/data_0.npz'
    args['image_loc'] = 'samples/denoised/denoised_pose/images/'
    visualise(args)
    args['frame'] = 'samples/denoised/initially_noisy/data_0.npz'
    args['image_loc'] = 'samples/denoised/initially_noisy/images/'
    visualise(args)


