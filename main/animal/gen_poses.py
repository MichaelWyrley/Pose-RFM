import sys
sys.path.append('')
import torch
import os
import numpy as np

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero_conditioned import DiT_adaLN_zero

from main.animal.vis_animal import visualise


def sample_grid_video(diffusion, labels,args):
    sample_full_for_video(diffusion, labels, args)
    for i in range(args['sample_timestep']):
        args['frame'] = args['sample_dir'] + f'sample_{i}.npz'
        args['name'] = "{:03d}".format(i)
        visualise(args)


def sample_grid(diffusion, labels, args):
    sample_single(diffusion, labels, args)
    visualise(args)

def sample_single(diffusion, labels,args):
    # Sample the full path for x ('samples') data points 
    if labels is None:
        sample = diffusion.sample(args['samples'], args['sample_timestep'], scale=args['scale'])
        pose_body = sample.cpu().numpy()

        np.savez(args['sample_dir'] + "sample_0", pose_body=pose_body, categories= np.zeros(args['samples'], dtype=int))
    else:

        sample = diffusion.sample(args['samples'], args['sample_timestep'], scale=args['scale'], labels=labels)
        pose_body = sample.cpu().numpy()
        labels = labels.cpu().numpy()

        np.savez(args['sample_dir'] + "sample_0", pose_body=pose_body, categories= labels)


def sample_full_for_video(diffusion, labels,args):
    # Sample the full path for x ('samples') data points 
    _, sample_list = diffusion.sample_full(args['samples'], args['sample_timestep'], scale=args['scale'], labels=labels)
    labels = labels.cpu().numpy()
    for i in range(args['sample_timestep']):
        sample = sample_list[i, :, :, :]
        pose_body = sample.cpu().numpy()

        np.savez(args['sample_dir'] + f"sample_{i}", pose_body=pose_body, categories= labels)
        if args['print']: print(f"Finished with {i}")


if __name__ == '__main__':
    args = {
        'samples': 25,
        'sample_timestep': 40,
        'load_model': 'best_model/animal_ema_model_800.pt',
        'scale': 6,
        'sample_dir': 'samples/animal_samples/',   
        'frame': 'samples/animal_samples/sample_0.npz',
        'image_loc': 'samples/animal_images/',

        'model': 'dataset/animal3d/MODELS/smpl_models/my_smpl_00781_4_all.pkl',
        'model_data': 'dataset/animal3d/MODELS/smpl_models/my_smpl_data_00781_4_all.pkl',
        'sym_file': 'dataset/animal3d/MODELS/smpl_models/symIdx.pkl',

        'print': True,
        'time_length': 15
        'output_obj': True,
        'name': '',
    }

    # os.mkdir(args['sample_dir'], exist_ok=True)
    # os.makedirs(args['image_loc'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero(in_dim=6, depth=16, emb_dimention=768, num_heads=16, num_classes=5).to(device)
    model.load_state_dict(torch.load(args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device, number_joints=34)

    labels = torch.Tensor([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]).to(int).to(device)
    # labels = torch.Tensor([0,0,0,1,1,1,4,4,4]).to(int).to(device)
    sample_grid_video(diffusion, labels, args)

