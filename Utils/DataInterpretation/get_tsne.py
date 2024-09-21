import sys
sys.path.append('')
import torch

from FlowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from VectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openTSNE


def sample_dataset(diffusion, no_samples, timestep = 35, scale = 3.5):
    _, sample_list = diffusion.sample_full(no_samples, timestep, scale=scale)
    return sample_list.reshape(sample_list.shape[0], sample_list.shape[1], -1).cpu().detach().numpy()

def reduce_data(data, tsne):
    full_data = []

    for i, d in enumerate(data):
        print("starting for timestep:", i)
        # reduced_data = tsne.fit_transform(d)
        reduced_data = tsne.fit(d)

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
    

def compute_heatmaps(diffusion, args):
    full_data = []

    # tsne = TSNE(n_components=2, perplexity = 20, early_exaggeration=12, n_iter_without_progress= 500, max_iter= 10000, method='exact')
    tsne = openTSNE.TSNE(
            perplexity=30,
            initialization="pca",
            metric="cosine",
            n_jobs=8,
            random_state=3,
        )

    for bs in range(0, args['samples'], args['batch_size']):
        print("starting batch", bs//args['batch_size'])
        data = sample_dataset(diffusion, args['batch_size'], args['sample_timestep'], args['scale'])
  
        full_data.append(data)

    full_data = np.concatenate(full_data, axis=1)
    full_data = reduce_data(full_data.reshape(full_data.shape[0], full_data.shape[1], -1), tsne)
    print(full_data.shape)
    gen_heatmap(full_data, args['save_loc'])

if __name__ == '__main__':
    args = {
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'sample_timestep': 35,
        'scale': 3.5,

        'samples': 1000,
        'batch_size': 250,

        'load_model': 'best_model/ema_model_1200.pt',
        'save_loc': 'samples/heatmaps'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = UNet(in_channels=15, out_channels=3, emb_dimention=256, img_size=32, num_heads=4, num_classes=10, condition_prob=0.25).to(device)
    model = DiT_adaLN_zero(in_dim=6, depth=12, emb_dimention=768, num_heads=12,).to(device)
    model.load_state_dict(torch.load(args['directory'] + args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    compute_heatmaps(diffusion, args)

