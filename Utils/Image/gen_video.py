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

def noise_pose(clean_pose, diffusion, noise_level=0.6):
    matrix_clean_pose = transforms.euler_angles_to_matrix(clean_pose, "XYZ")
    noise = diffusion.gen_random_x(matrix_clean_pose)

    timestep = torch.full((clean_pose.shape[0],1), noise_level, device=clean_pose.device)

    noisy_pose = diffusion.conditional_flow(noise, matrix_clean_pose, timestep[:, None])

    return transforms.matrix_to_axis_angle(noisy_pose)


def gen_video(diffusion, model, initial_value, args):

    # Sample the full path for x ('samples') data points 

    frames = torch.zeros(args['no_frames'], 1, 21,3).to(model.device)
    frames[0] = initial_value

    with torch.no_grad():
        for i in range(1, args['no_frames']):
            noisy_pose = noise_pose(frames[i-1], diffusion, noise_level=args['initial_timestep'] / args['timesteps'])

            sampled = diffusion.denoise_pose(noisy_pose, args['initial_timestep'], args['timesteps'], scale=args['scale'])

            frames[i] = sampled[0]
            
        
        np.savez(args['directory'] + args['frame'], pose_body=frames.cpu().numpy().reshape(-1,21,3))
    

if __name__ == '__main__':
    args = {
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'clean': './dataset/amass/SAMPLED_POSES/',

        'scale': 0.25,
        'no_frames': 180,

        'initial_timestep': 99,
        'timesteps': 100,
        # Make 0 if don't want to use
        'load_model': 'models/model_300.pt',
        'sample_dir': 'samples/sample_list/',

        'frame': 'samples/gen_video/data_0.npz',
        'image_loc': 'samples/gen_video/images/',
        'model': './dataset/models/neutral/model.npz',

        'initial_pose': 'samples/sample_list/sample_49.npz',

        'name': '',
        'print': False,
        'save_grid': False,
        'sample_single': True

    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero().to(device)
    model.load_state_dict(torch.load(args['directory'] + args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    bdata = np.load(args['initial_pose'])
    initial_pose = torch.Tensor(bdata['pose_body'][:1]).to(device)

    gen_video(diffusion, model, initial_pose, args)
    visualise(args)


