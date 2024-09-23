import sys
sys.path.append('')
import torch
import os

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero
from main.utils.NRDF.data.dataloaders import PoseData

from main.utils.image.visualise import visualise
import numpy as np

def noise_pose(clean_pose, noise_level=0.1):
    noise = torch.randn_like(clean_pose)
    noisy_pose = clean_pose + noise * noise_level

    return noisy_pose


def denoise(diffusion, args):

    data_set = PoseData(mode='train', clean_dir=args['clean'], batch_size=args['batch_size'], num_workers=6, num_pts=args['no_samples'], stage=1, flip=False, random_poses=False)
    data_loader = data_set.get_loader()

    with torch.no_grad():
        for i, poses in enumerate(data_loader):
            clean_pose = poses.to(device).reshape(-1, 21, 3)
            noisy_pose = noise_pose(clean_pose, noise_level=args['noise_level'])

            sampled = diffusion.denoise_pose(noisy_pose, args['initial_timestep'], args['timesteps'], scale=args['scale'])

            denoised_pose = sampled.cpu().numpy()
            clean_pose = clean_pose.cpu().numpy()
            noisy_pose = noisy_pose.cpu().numpy()
            np.savez(f'./samples/denoised/denoised_pose/data_{i}.npz', pose_body=denoised_pose)
            np.savez(f'./samples/denoised/clean_pose/data_{i}.npz', pose_body=clean_pose)
            np.savez(f'./samples/denoised/initially_noisy/data_{i}.npz', pose_body=noisy_pose)
            break
    

if __name__ == '__main__':
    args = {
        'clean': './dataset/amass/SAMPLED_POSES/',

        'batch_size': 1,
        'no_samples': 16,
        'scale': 2,
        'noise_level': 0.2,

        'initial_timestep': 90,
        'timesteps': 100,
        'load_model': 'models/ema_model_400.pt',

        'frame': 'samples/denoised/clean_pose/data_0.npz',
        'image_loc': 'samples/denoised/clean_pose/images/',
        'model': './dataset/models/neutral/model.npz',

        'name': '',
        'print': False,
        'save_grid': True,

    }

    os.mkdir('samples/denoised/denoised_pose', exist_ok=True)
    os.mkdir('samples/denoised/clean_pose', exist_ok=True)
    os.mkdir('samples/denoised/initially_noisy', exist_ok=True)
    os.mkdir('samples/denoised/denoised_pose/images', exist_ok=True)
    os.mkdir('samples/denoised/clean_pose/images', exist_ok=True)
    os.mkdir('samples/denoised/initially_noisy/images', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero().to(device)
    model.load_state_dict(torch.load(args['load_model']))
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


