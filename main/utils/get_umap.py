import sys
sys.path.append('')
import torch

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import openTSNE
from cuml.manifold import UMAP
from cuml.common.device_selection import using_device_type

def sample_dataset(diffusion, no_samples, timestep = 35, scale = 3.5):
    _, sample_list = diffusion.sample_full(no_samples, timestep, scale=scale)
    return sample_list.reshape(sample_list.shape[0], sample_list.shape[1], -1).cpu().detach().numpy()

def reduce_data(data, umap):
    full_data = []

    for i, d in enumerate(data):
        print("starting for timestep:", i)

        with using_device_type('gpu'):
            reduced_data = umap.fit_transform(d)


        full_data.append(reduced_data)

    return np.array(full_data)


def gen_heatmap(data, save_loc):
    np.savez(save_loc+"/data", data)

    plt.figure(figsize=(20, 20), dpi=80)
    
    for i in range(data.shape[0]):
        print("saving denoise step {:03d}".format(i))

        sns.set_style("white")
        sns.kdeplot(x=data[i,:,0], y=data[i,:,1])
        plt.axis('off')
        plt.savefig(save_loc +"/{:03d}.png".format(i), bbox_inches='tight')
        plt.clf()

def load_heatmap(save_loc): 
    data = np.load(save_loc + "/data.npz")['arr_0']

    plt.figure(figsize=(20, 20), dpi=80)
    
    for i in range(data.shape[0]):
        print("saving denoise step {:03d}".format(i))

        sns.set_style("white")
        sns.kdeplot(x=data[i,:,0], y=data[i,:,1], fill=False, thresh=0, levels=20)
        plt.axis('off')
        plt.savefig(save_loc +"/{:03d}.png".format(i), bbox_inches='tight')
        plt.clf()

def compute_heatmaps(diffusion, args):
    full_data = []

    umap = UMAP(     
        n_components=2,   
        n_neighbors=80,
        min_dist=0.1,
        metric="cosine",
        angular_rp_forest=True,
        random_state=42)

    for bs in range(0, args['samples'], args['batch_size']):
        print("starting batch", bs//args['batch_size'])
        data = sample_dataset(diffusion, args['batch_size'], args['sample_timestep'], args['scale'])
  
        full_data.append(data)

    full_data = np.concatenate(full_data, axis=1)
    full_data = reduce_data(full_data.reshape(full_data.shape[0], full_data.shape[1], -1), umap)

    gen_heatmap(full_data, args['save_loc'])

if __name__ == '__main__':
    args = {
        'sample_timestep': 35,
        'scale': 3.8,

        'samples': 10000,
        'batch_size': 500,

        'load_model': 'best_model/ema_model_1200.pt',
        'save_loc': 'samples/heatmaps'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = DiT_adaLN_zero(in_dim=6, depth=12, emb_dimention=768, num_heads=12,).to(device)
    model.load_state_dict(torch.load(args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)
    compute_heatmaps(diffusion, args)

