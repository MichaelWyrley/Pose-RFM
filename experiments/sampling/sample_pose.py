import os
import sys
# add the current working directory so this can be run from the github repo root !!
sys.path.append(os.getcwd())

import torch
import matplotlib.pyplot as plt

from FlowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from VectorFieldModels.Transformer import DiT
from Utils.NRDF.data.dataloaders import PoseData
from VectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero

from Utils.NRDF.utils.transforms import quaternion_to_axis_angle
import numpy as np

def sample_model(diffusion, args):
    # Sample the full path for x ('samples') data points 
    for i in range(args['no_samples']):
        sampled = diffusion.sample(args['samples'], args['sample_timestep'], scale=args['scale'])
        pose_body = sampled.cpu().numpy()
        np.savez(args['directory'] + args['save_location'] + f'{i}.npz', pose_body=pose_body)
        print(f"done sampling: {i}" )
    
def sample_dataset(args):
    pass

if __name__ == '__main__':
    args = {
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'save_location': 'experiments/samples/generated_samples/', 
        'samples': 500,
        'no_samples': 20,
        'sample_timestep': 35,
        'scale': 3,
        # Make 0 if don't want to use
        'load_model': 'models/ema_model_350.pt',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero().to(device)
    model.load_state_dict(torch.load(args['directory'] + args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    sample_model(diffusion,args)


