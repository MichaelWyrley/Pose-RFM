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
from cuml import PCA, TruncatedSVD, IncrementalPCA

def sample_dataset(diffusion, no_samples, timestep = 35, scale = 3.5):
    _, sample_list = diffusion.sample_full(no_samples, timestep, scale=scale)
    return sample_list.reshape(sample_list.shape[0], sample_list.shape[1], -1).cpu().detach().numpy()

# def remove_similar_poses(data, threshold=0.1):
#     print("removing similar poses")
#     new_data = []

#     for i in range(data.shape[0]):
#         for j in range(i+1, data.shape[0]):
#             distance = np.mean(np.linalg.norm(data[i] - data[j], dim = 1))
            

#     return np.array(new_data)

def reduce_data(data, umap):
    full_data = []

    with using_device_type('gpu'):
        reduced_data = umap.fit_transform(data[0])
        full_data.append(reduced_data)

    for i, d in enumerate(data[1:]):
        print("starting for timestep:", i)

        with using_device_type('gpu'):
            reduced_data = umap.fit_transform(d)
            full_data.append(reduced_data)

    return np.array(full_data)

def gen_single_bone(data, save_loc):
    plt.figure(figsize=(20, 20), dpi=80)
    
    for i in range(data.shape[0]):
        print("saving denoise step {:03d}".format(i))

        sns.set_style("white")
        sns.kdeplot(x=data[i,:,0], y=data[i,:,1])
        plt.axis('off')
        plt.savefig(save_loc +"/{:03d}.png".format(i), bbox_inches='tight')
        plt.clf()

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
    # Docs https://docs.rapids.ai/api/cuml/stable/api/#umap
    umap = UMAP(     
        n_neighbors=100,
        n_components=2,   
        n_neighbors=80,
        min_dist=0.1,
        metric="cosine",
        angular_rp_forest=True,
        random_state=42)
        # min_dist=0.05,
        # local_connectivity=2,
        # negative_sample_rate=4,
        # hash_input=True,
        # angular_rp_forest=True)
    
    # pca = PCA(n_components=2)
    # pca = TruncatedSVD(n_components=2)
    pca = IncrementalPCA(n_components=2, whiten=True)

    for bs in range(0, args['samples'], args['batch_size']):
        print("starting batch", bs//args['batch_size'])
        data = sample_dataset(diffusion, args['batch_size'], args['sample_timestep'], args['scale'])
  
        full_data.append(data)

    full_data = np.concatenate(full_data, axis=1)
    print(full_data.shape)
    np.savez(args['save_loc']+"/../data", poses=full_data)

    # gen_single_bone(full_data.reshape(full_data.shape[0], full_data.shape[1], -1), args['save_bone_loc'])

    # full_data_umap = reduce_data(full_data.reshape(full_data.shape[0], full_data.shape[1], -1), umap)
    # full_data_pca = reduce_data(full_data.reshape(full_data.shape[0], full_data.shape[1], -1), pca)
    # gen_heatmap(full_data_umap, args['save_loc'])
    # gen_heatmap(full_data_pca, args['save_loc_pca'])

if __name__ == '__main__':
    args = {
        'sample_timestep': 25,
        'scale': 3.8,

        'samples': 5000,
        'batch_size': 500,

        'load_model': 'best_model/ema_model_1200.pt',
        'save_loc': 'samples/heatmaps',
        'save_loc_pca': 'samples/heatmaps_int_pca',
        'save_bone_loc': 'samples/bone',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = DiT_adaLN_zero(in_dim=6, depth=12, emb_dimention=768, num_heads=12,).to(device)
    model.load_state_dict(torch.load(args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)
    compute_heatmaps(diffusion, args)

