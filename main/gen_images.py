import sys
sys.path.append('')
import torch
from main.utils.sampling.sample import sample_single, sample_full, sample_full_for_video
# from main.utils.image.visualise import visualise
from main.utils.image.visualise_torch3d import visualise

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero

import os

# Sample and generate images for the final poses from the model 
def sample_grid(diffusion, args):
    # sample_single(diffusion, args)
    visualise(args)

# Sample and generate images for the full path poses from the model 
# This saves and generates images for the full path for each generated samples
def sample_video(diffusion, args):
    sample_full(diffusion, args)
    visualise(args)

# Sample and generate images for the full path poses from the model 
# This saves and generates images for each timestep in all poses generated
def sample_grid_video(diffusion, args):
    sample_full_for_video(diffusion, args)
    for i in range(args['sample_timestep']):
        args['frame'] = f'samples/sample_list/sample_{i}.npz'
        args['name'] = "{:03d}".format(i)
        visualise(args)


if __name__ == '__main__':

    args = {
        'samples': 16,
        'sample_timestep': 25,
        'load_model': 'best_model/ema_model_1200.pt',
        'scale': 3.8,
        'sample_dir': 'samples/sample_list/',
        'name': '',
     
        'frame': 'samples/sample_list/sample_0.npz',
        'model': '../individual-project/dataset/models/neutral/model.npz',
        'image_loc': './samples/images/',
        'print': True,

        'output_obj': True,
        'time_length': 10000,

        'save_grid': True,
    }

    # os.mkdirs(args['sample_dir'], exist_ok=True)
    # os.mkdir(args['image_loc'], exist_ok=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = DiT_adaLN_zero(in_dim=6, depth=12, emb_dimention=768, num_heads=12,).to(device)
    model.load_state_dict(torch.load(args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    sample_grid(diffusion, args)

