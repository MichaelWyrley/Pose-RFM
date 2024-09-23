import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
# add the current working directory so this can be run from the github repo root !!
sys.path.append(os.getcwd())
import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments.tests.test_generation import average_pairwise_distance, frechet_distance
from sampling.sample_pose import sample_model
from experiments.tests.test_pose_denoising import project_poses

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero
from main.vectorFieldModels.Transformer import DiT

def test_denoise_different_models():
    print("----------test_different_models---------")
    args = {
        'no_samples': 20,
        
        'dataset_directory': './dataset/amass/SAMPLED_POSES/',
        'generated_directory': 'experiments/samples/generated_samples/',
        'model': './dataset/models/neutral/model.npz',
        'dataset_name': 'pose_body',
        'dataset_size': 500,
        'dataset_data': 'experiments/utils/dataset_mean_cov.npz',

        # 'load_model': ['working_models/noised_pose_50/ema_model_400.pt','working_models/noised_pose_75/ema_model_400.pt','working_models/noised_pose_90/ema_model_400.pt','working_models/noised_pose_100/ema_model_400.pt'],
        # 'load_model': ['models/model_900.pt', 'models/model_1000.pt', 'models/model_1100.pt', 'models/model_1200.pt'],
        'load_model': ['working_models/noised_pose_0/ema_model_400.pt'],

        'file': 'experiments/current_tests/test_denoising_different_models.txt',

        'noisy': 'dataset/amass/NOISY_POSES/examples/noisy_pose.npz',

        'save_location': 'experiments/samples/denoised_pose/',
    
        'faiss_model': 'dataset/amass/FAISS_MODEL',
        'k-faiss': 1000,
        'k-dist': 1,

        'batch_size': 500,
        'initial_timestep': 5,
        'timesteps': 15,
        'scale': 4,

        'no_samples': 1,

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    with open(args['file'], 'a') as f:
        for i in args['load_model']:
            print("testing model", i)
            model = DiT_adaLN_zero().to(device)
            model.load_state_dict(torch.load(i))
            model.eval()

            diffusion = FlowMatchingMatrix(model, device=device)
            
            geo_m2m_mean, geo_m2m_std, m2m_dist_mean, m2m_dist_std = project_poses(diffusion, args)

            f.write("MODEL = " + i + "\n")
            f.write(str(geo_m2m_mean) + ',' + str(geo_m2m_std) + ',' + str(m2m_dist_mean) + ',' +str(m2m_dist_std) + '\n')

def test_different_models():
    print("----------test_different_models---------")
    args = {
        'save_location': 'experiments/samples/generated_samples/', 
        'samples': 500,
        'no_samples': 20,
        # Change to 25 to match timestep thing from flow matching
        'sample_timestep': 10,
        'scale': 3.5,
        

        'dataset_directory': './dataset/amass/SAMPLED_POSES/',
        'generated_directory': 'experiments/samples/generated_samples/',
        'model': './dataset/models/neutral/model.npz',
        'dataset_name': 'pose_body',
        'dataset_size': 500,
        'dataset_data': 'experiments/utils/dataset_mean_cov.npz',

        # 'load_model': ['working_models/time_0/ema_model_400.pt','working_models/noised_pose_90/ema_model_400.pt','working_models/time_50/ema_model_400.pt','working_models/time_75/ema_model_400.pt','working_models/time_100/ema_model_400.pt'],
        # 'load_model': ['models/model_900.pt', 'models/model_1000.pt', 'models/model_1100.pt', 'models/model_1200.pt'],
        'load_model': ['working_models/time_sample_10/model_400.pt'],
        
        'file': 'experiments/current_tests/test_different_models_time_samples.txt'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    with open(args['file'], 'a') as f:
        for i in args['load_model']:
            print("testing model", i)
            model = DiT_adaLN_zero().to(device)
            model.load_state_dict(torch.load(i))
            model.eval()

            diffusion = FlowMatchingMatrix(model, device=device)
            
            sample_model(diffusion, args)
            
            apd_mean, apd_std = average_pairwise_distance(args)
            fd_mean, fd_std = frechet_distance(args)

            f.write("MODEL = " + i + "\n")
            f.write(str(apd_mean) + ',' + str(apd_std) + ',' + str(fd_mean) + ',' +str(fd_std) + "\n")

def test_dit_vs_ada_dit():
    print("----------test_different_models---------")
    args = {
        'save_location': 'experiments/samples/generated_samples/', 
        'samples': 500,
        'no_samples': 20,
        'sample_timestep': 30,
        'scale': 3.5,
        

        'dataset_directory': './dataset/amass/SAMPLED_POSES/',
        'generated_directory': 'experiments/samples/generated_samples/',
        'model': './dataset/models/neutral/model.npz',
        'dataset_name': 'pose_body',
        'dataset_size': 500,
        'dataset_data': 'experiments/utils/dataset_mean_cov.npz',

        'load_model': ['working_models/noised_pose_90/model_400.pt', 'working_models/DiT/model_400.pt'],
        # 'load_model': ['models/model_900.pt', 'models/model_1000.pt', 'models/model_1100.pt', 'models/model_1200.pt'],
        
        'file': 'experiments/test_different_models.txt'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    with open(args['file'], 'a') as f:
        print("testing model dit")
        model = DiT().to(device)
        model.load_state_dict(torch.load(args['load_model'][1]))
        model.eval()
        diffusion = FlowMatchingMatrix(model, device=device)
        
        sample_model(diffusion, args)
        
        apd_mean, apd_std = average_pairwise_distance(args)
        fd_mean, fd_std = frechet_distance(args)

        f.write("MODEL = " + args['load_model'][1] + "\n")
        f.write(str(apd_mean) + ',' + str(apd_std) + ',' + str(fd_mean) + ',' +str(fd_std) + "\n")


def test_differnt_no_samples_and_scales():
    print("----------test_differnt_no_samples_and_scales---------")
    args = {
        'save_location': 'experiments/samples/generated_samples/', 
        'samples': 500,
        'no_samples': 20,
        # next do scale 3, for 1,5,10,20,25,30,35,40,50
        'all_scale': [3.5],
        'all_sample_timestep': [1],

        'scale': -1,
        'sample_timestep': -1,

        'dataset_directory': './dataset/amass/SAMPLED_POSES/',
        'generated_directory': 'experiments/samples/generated_samples/',
        'model': './dataset/models/neutral/model.npz',
        'dataset_name': 'pose_body',
        'dataset_size': 500,
        'dataset_data': 'experiments/utils/dataset_mean_cov.npz',

        'load_model': ['working_models/time_sample_5/model_400.pt', 'working_models/time_sample_10/model_400.pt', 'working_models/time_sample_15/model_400.pt', 'working_models/time_sample_25/model_400.pt', 'working_models/noised_pose_90/model_400.pt'],
        'file': 'experiments/test_differnt_no_samples_and_scales_time.txt'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    with open(args['file'], 'a') as f:
        for model_loc in args['load_model']:

            model = DiT_adaLN_zero().to(device)
            model.load_state_dict(torch.load(model_loc))
            model.eval()

            diffusion = FlowMatchingMatrix(model, device=device)
            f.write('-----------------------------------------\n')
            f.write(f'{model_loc}\n')
            for scale in args['all_scale']:
                for timesteps in args['all_sample_timestep']:

                    args['scale'] = scale
                    args['sample_timestep'] = timesteps

                    sample_model(diffusion, args)
                    
                    apd_mean, apd_std = average_pairwise_distance(args)
                    fd_mean, fd_std = frechet_distance(args)

                    f.write("timestep=" + str(timesteps) + ",scale=" + str(scale) + "\n")
                    f.write(str(apd_mean) + ',' + str(apd_std) + ',' + str(fd_mean) + ',' +str(fd_std) + "\n")



if __name__ == '__main__':
    # test_different_models()
    test_differnt_no_samples_and_scales()
    # test_dit_vs_ada_dit()
    # test_denoise_different_models()


