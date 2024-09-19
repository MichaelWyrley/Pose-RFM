import pickle
import glob
import numpy as np
import os

def unpickle(args):
    gt_directories = sorted(glob.glob(args['directory'] + args['data_files']))
    # gen_directories = sorted(glob.glob(args['directory'] + args['gen_pose_files']))


    for gt_dir in gt_directories:
        name = gt_dir.split('/')[-1].split('.')[0]

        with open(gt_dir, "rb") as f:
            item = pickle.load(f, encoding='latin1')
            np.savez(args['pkl_save_loc'] + '/' + name, ** item)

# def check_files(args):
#     data_files = sorted(glob.glob(args['directory'] + args['data_files']))
#     img_files = sorted(os.listdir((args['directory'] + args['img_files'])))
    
#     print(len(data_files), len(img_files))

#     for i in zip(data_files, img_files):
#         print(i[0].split('/')[-1].split('.')[0], i[1].split('/')[-1].split('.')[0])
    
# def fix_smplx_file(args):
#     ground_truth_files = sorted(glob.glob(args['ground_truth_files'] + '*.npz'))
    
#     generated_files = sorted(glob.glob(args['gen_pose_files']))
#     generated_files = [ds for ds in generated_files if args['ground_truth_files'] + ds.split('/')[-1] in ground_truth_files]


#     for (gtf, genf) in zip(ground_truth_files, generated_files):
#         name = genf.split('/')[-1].split('.')[0]
#         ground_truth_sequence = np.load(gtf, allow_pickle=True)
#         generated_sequence = np.load(genf, allow_pickle=True)
#         gtf_poses = np.array(ground_truth_sequence['pose_body']) # 3:72 then :63
#         gtf_betas = np.array(ground_truth_sequence['betas'])

#         new_gft_poses = []
#         new_genf_poses = []
#         new_genf_betas = []
#         for gtf_pose, gen_pose in zip(gtf_poses, generated_sequence['pose_body']):
#             try:
#                 gen_pose = np.array(gen_pose[0])
#                 gen_betas = np.array(gen_pose[0])
#                 if gen_pose.shape == (21, 3):
#                     new_gft_poses.append(gtf_pose)
#                     new_genf_poses.append(gen_pose)
#                     new_genf_betas.append(gen_betas)
#                 else:
#                     print("not", gen_pose)
#             except:
#                 print("not", gen_pose)

                
#         print(len(new_gft_poses), len(new_genf_poses), len(new_genf_betas))

#         np.savez(args['directory'] + args['fix_save_loc'] + name, pose_body=np.array(new_genf_poses), betas = np.array(new_genf_betas))
#         np.savez(args['directory'] + args['pkl_save_loc'] + name, pose_body = np.array(new_gft_poses), betas = np.array(gtf_betas))

if __name__ == '__main__':
    args = {
        # 'support_dir': '/vol/bitbucket/mew23/individual_project/',
        'directory': '/vol/bitbucket/mew23/individual-project/',
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