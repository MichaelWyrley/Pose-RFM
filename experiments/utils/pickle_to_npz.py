import pickle
import glob
import numpy as np
import os

def unpickle(args):
    gt_directories = sorted(glob.glob(args['data_files']))

    for gt_dir in gt_directories:
        name = gt_dir.split('/')[-1].split('.')[0]

        with open(gt_dir, "rb") as f:
            item = pickle.load(f, encoding='latin1')
            np.savez(args['pkl_save_loc'] + '/' + name, ** item)

if __name__ == '__main__':
    args = {
        # 'support_dir': '/vol/bitbucket/mew23/individual_project/',
        'data_files': 'dataset/3DPW/sequenceFiles/test/*.pkl',
        'gen_pose_files': 'dataset/3DPW/smplx_poses/*.npz',
        'ground_truth_files': 'dataset/3DPW/npz_poses/ground_truth/',
        'img_files': 'dataset/3DPW/imageFiles/test/',
        'pkl_save_loc': 'dataset/3DPW/npz_poses/ground_truth/',
        'fix_save_loc': 'dataset/3DPW/npz_poses/smplx/'
    }

    unpickle(args)
    # check_files(args)
    # fix_smplx_file(args)