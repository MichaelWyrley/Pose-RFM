import sys
sys.path.append('')
import torch
from Utils.Image.sample import sample_single, sample_full, sample_full_for_video
from Utils.Image.visualise import visualise

from FlowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
# from FlowMatchingModels.motionFlowMatchingMatrix import FlowMatchingMatrix
from VectorFieldModels.Transformer import DiT
from VectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero

def sample_grid(diffusion, args):
    sample_single(diffusion, args)
    visualise(args)


def sample_video(diffusion, args):
    sample_full(diffusion, args)
    visualise(args)

def sample_grid_video(diffusion, args):
    sample_full_for_video(diffusion, args)
    for i in range(args['sample_timestep']):
        args['frame'] = f'samples/sample_list/sample_{i}.npz'
        args['name'] = "{:03d}".format(i)
        visualise(args)


if __name__ == '__main__':
    args = {
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'samples': 16,
        'sample_timestep': 35,
        'load_model': 'best_model/ema_model_1200.pt',
        'scale': 4,
        'sample_dir': 'samples/sample_list/',
        'name': '',
     
        'frame': 'samples/sample_list/sample_0.npz',
        'model': './dataset/models/neutral/model.npz',
        'image_loc': './',
        'print': True,

        'save_grid': True,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero(in_dim=6, depth=12, emb_dimention=768, num_heads=12,).to(device)
    model.load_state_dict(torch.load(args['directory'] + args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    sample_grid(diffusion, args)

