import sys
sys.path.append('')
import torch

from flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from vectorFieldModels.Transformer import DiT_adaLN_zero

from utils.NRDF.utils.transforms import quaternion_to_axis_angle
import numpy as np
import os

# Sample the full trajectory of all samples, then save them as individual samples
def sample_full(diffusion, args):
    # Sample the full path for x ('samples') data points 
    _, sample_list = diffusion.sample_full(args['samples'], args['sample_timestep'], scale=args['scale'])
    for i in range(args['samples']):
        sample = sample_list[:, i, :, :]
        pose_body = sample.cpu().numpy()

        np.savez(args['sample_dir'] + f"sample_{i}", pose_body=pose_body)
        if args['print']: print(f"Finished with {i}")

# Sample the full trajectory of all samples, then save them as individual timesteps
def sample_full_for_video(diffusion,args):
    # Sample the full path for x ('samples') data points 
    _, sample_list = diffusion.sample_full(args['samples'], args['sample_timestep'], scale=args['scale'])
    for i in range(args['sample_timestep']):
        sample = sample_list[i, :, :, :]
        pose_body = sample.cpu().numpy()

        np.savez(args['sample_dir'] + f"sample_{i}", pose_body=pose_body)
        if args['print']: print(f"Finished with {i}")

# Sample the final pose of all samples, then save them as one file
def sample_single(diffusion,args):
    # Sample the full path for x ('samples') data points 
    sample = diffusion.sample(args['samples'], args['sample_timestep'], scale=args['scale'])
    pose_body = sample.cpu().numpy()

    np.savez(args['sample_dir'] + f"sample_0", pose_body=pose_body)

if __name__ == '__main__':
    args = {
        'samples': 16,
        'sample_timestep': 52,
        'print': True,
        'load_model': 'models/model_700.pt',
        'scale': 12,
        'sample_dir': 'samples/sample_list/'
    }

    os.mkdir(args['sample_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero().to(device)
    model.load_state_dict(torch.load(args['directory'] + args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    sample_single(args)

