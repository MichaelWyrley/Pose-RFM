import sys
sys.path.append('')

import glob
import numpy as np

from main.utils.NRDF.utils.data_utils import amass_splits

def gen_beta_information(args):
    dataset = sorted(glob.glob(args['data_files'] + '/*/*'))
    dataset = [ds for ds in dataset if ds.split('/')[-2] in amass_splits['train']]

    betas_list = []
    for file in dataset:
        print('processing file ', file.split('/')[-2], file.split('/')[-1])
        npzs = sorted(glob.glob(file + '/*.npz'))
        if len(npzs) == 0:
            continue
        
        data = np.load(npzs[0])
        betas = data['betas']

        betas_list.append(betas)
    
    betas_list = np.array(betas_list)[...,:10]
    print(betas_list.shape)

    mean_betas = betas_list.mean(axis=0)
    betas_cov = np.cov(betas_list, rowvar=False)

    print(mean_betas.shape, betas_cov.shape)

    np.savez(args['save_loc'], betas=betas_list, mean_betas = mean_betas, betas_cov = betas_cov)

if __name__ == '__main__':
    args = {
        # 'support_dir': '/vol/bitbucket/mew23/individual_project/',
        'data_files': 'dataset/amass/RAW_DATA',
        'save_loc': './experiments/utils/betas'
    }

    gen_beta_information(args)
