import sys
sys.path.append('')
import torch
import os
import matplotlib.pyplot as plt

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero

from main.utils.NRDF.data.dataloaders import PoseData
import pytorch3d.transforms as transforms
from main.utils.image.visualise import visualise
import numpy as np

def partial_corruption(clean_pose, diffusion, removal_level=0.6):
    matrix_clean_pose = transforms.euler_angles_to_matrix(clean_pose, "XYZ")
    noise = diffusion.gen_random_x(matrix_clean_pose)

    sample_removal = torch.rand((matrix_clean_pose.shape[0], matrix_clean_pose.shape[1]))
    sample_removal = sample_removal > removal_level
    
    matrix_clean_pose[~ sample_removal] = noise[~ sample_removal]
    matrix_clean_pose = transforms.matrix_to_axis_angle(matrix_clean_pose)

    return matrix_clean_pose, sample_removal


def partial_generation(diffusion, args):

    data_set = PoseData(mode='train', clean_dir=args['clean'], batch_size=args['batch_size'], num_workers=6, num_pts=args['no_samples'], stage=1, flip=False, random_poses=False)
    data_loader = data_set.get_loader()

    with torch.no_grad():
        for i, poses in enumerate(data_loader):
            clean_pose = poses.to(device).reshape(-1, 21, 3)
            partial_pose, mask = partial_corruption(clean_pose, diffusion, removal_level=args['removal_level'])

            sampled = diffusion.sample_partial(partial_pose, mask, args['timesteps'], scale=args['scale'])

            partial_pose = sampled.cpu().numpy()
            clean_pose = clean_pose.cpu().numpy()
            np.savez(f'./samples/missing_points/generated/data_{i}.npz', pose_body=partial_pose)
            np.savez(f'./samples/missing_points/original/data_{i}.npz', pose_body=clean_pose)
            break
    

if __name__ == '__main__':
    args = {
        'clean': './dataset/amass/SAMPLED_POSES/',

        'batch_size': 1,
        'no_samples': 16,
        'scale': 4,

        'removal_level': 0.2,
        'timesteps': 50,
        # Make 0 if don't want to use
        'load_model': 'models/ema_model_400.pt',

        'frame': 'samples/missing_points/original/data_0.npz',
        'image_loc': 'samples/missing_points/original/images/',
        'model': './dataset/models/neutral/model.npz',

        'name': '',
        'print': False,
        'save_grid': True,
        'sample_single': True

    }

    os.mkdir('samples/missing_points/original', exist_ok=True)
    os.mkdir('samples/missing_points/generated', exist_ok=True)
    os.mkdir('samples/missing_points/original/images', exist_ok=True)
    os.mkdir('samples/missing_points/generated/images', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero().to(device)
    model.load_state_dict(torch.load(args['directory'] + args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    partial_generation(diffusion, args)
    visualise(args)
    args['frame'] = 'samples/missing_points/generated/data_0.npz'
    args['image_loc'] = 'samples/missing_points/generated/images/'
    visualise(args)


