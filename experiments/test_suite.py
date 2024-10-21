import sys
sys.path.append('')
# add the current working directory so this can be run from the github repo root !!
# sys.path.append(os.getcwd())
import torch

from experiments.tests.test_generation import average_pairwise_distance, frechet_distance, distance_gen_dataset
from experiments.sampling.sample_pose import sample_model
from experiments.tests.test_partial_generation import gen_partial
from experiments.tests.test_pose_denoising import project_poses

from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero
import numpy as np

def confidence_interval(std, sample_size, z_value = 	1.960):
    return z_value * std / np.sqrt(sample_size)


def generic_generaion(model):
    print("----------GENERATING GENERIC INFORMATION---------")
    args = {
        'samples': 500,
        'no_samples': 20,
        'sample_timestep': 30,
        'scale': 3.5,
        'dataset_data': 'experiments/utils/dataset_mean_cov.npz',
        'save_location': 'experiments/samples/generated_samples/', 

        'faiss_model': 'dataset/amass/FAISS_MODEL',
        'k-faiss': 1000,
        'k-dist': 1,

        'dataset_directory': 'dataset/amass/SAMPLED_POSES/',
        'generated_directory': 'experiments/samples/generated_samples/',
        'model': 'dataset/models/neutral/model.npz',
        'dataset_name': 'pose_body',
        'dataset_size': 500,
    }
    
    sample_model(model, args)
    
    apd_mean, apd_std = average_pairwise_distance(args)
    nn_mean, nn_std = distance_gen_dataset(args)
    fd_mean, fd_std = frechet_distance(args)

    confidence_interval_apd = confidence_interval(apd_std, args['no_samples'])
    confidence_interval_nn = confidence_interval(nn_std, args['no_samples'])
    confidence_interval_fd = confidence_interval(fd_std, args['no_samples'])

    return apd_mean, apd_std, confidence_interval_apd, nn_mean, nn_std, confidence_interval_nn, fd_mean, fd_std, confidence_interval_fd

def partial_generation(model):
    print("----------GENERATING PARTIAL INFORMATION---------")
    args = {
        'clean': 'dataset/amass/SAMPLED_POSES/',
        'model': 'dataset/models/neutral/model.npz',

        'save_location': 'experiments/samples/partial_generation/', 
        'scale': 3.5,
        'timesteps': 15,
        'removal_level': 0.2,
        'no_masks': 50, 
        'batch_size': 500,
        'gen_masks': True,
        'dataset_data': 'experiments/utils/dataset_mean_cov.npz',
        'random_subset': 50,

        'stop_sampling': 1,

        'no_samples': 20,

        'faiss_model': 'dataset/amass/FAISS_MODEL',
    }

    apd_mean, apd_std, dnn_mean, dnn_std, fd_mean, fd_std = gen_partial(model, args)

    confidence_interval_apd = confidence_interval(apd_std, args['no_samples'])
    confidence_interval_nn = confidence_interval(dnn_std, args['no_samples'])
    confidence_interval_fd = confidence_interval(fd_std, args['no_samples'])

    return apd_mean, apd_std, confidence_interval_apd, nn_mean, nn_std, confidence_interval_nn, fd_mean, fd_std, confidence_interval_fd


def denoising_generation(model):
    print("-----------GENERATING NOISY POSES-----------")
    args = {
        # 'support_dir': '/vol/bitbucket/mew23/individual_project/',
        'noisy': 'dataset/amass/NOISY_POSES/examples/noisy_pose.npz',

        'model': 'dataset/models/neutral/model.npz',

        'faiss_model': 'dataset/amass/FAISS_MODEL',
        'k-faiss': 1000,
        'k-dist': 1,

        'dataset_name': 'pose_body',

        'batch_size': 500,
        'initial_timestep': 10,
        'timesteps': 15,
        'scale': 4,

        'no_samples': 20,

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    geo_m2m_mean, geo_m2m_std, m2m_dist_mean, m2m_dist_std = project_poses(model, args)
    return geo_m2m_mean, geo_m2m_std, m2m_dist_mean, m2m_dist_std


if __name__ == '__main__':
    args = {
        'load_model': 'samples/training_models/ema_model_1200.pt',
        'file': 'experiments/current_tests/test_suit.txt'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = DiT_adaLN_zero().to(device)
    model.load_state_dict(torch.load(args['load_model']))
    model.eval()

    diffusion = FlowMatchingMatrix(model, device=device)

    with open(args['file'], 'a') as f:
        # f.write("\n\nMODEL = " + args['load_model'] + "\n")
        # apd_mean, apd_std, conv_inv_apd, nn_mean, nn_std, conv_inv_nn, fd_mean, fd_std, conv_inv_fd = generic_generaion(diffusion)
        # f.write('#Generic Generation\n')
        # f.write(str(apd_mean) + ',' + str(apd_std) + ',' + str(conv_inv_apd) + ',' +str(nn_mean) + ',' + str(nn_std)+ ',' + str(conv_inv_nn) + ',' + str(fd_mean) + ',' +str(fd_std)+ ',' + str(conv_inv_fd) + "\n")
        # apd_mean, apd_std, conv_inv_apd, nn_mean, nn_std, conv_inv_nn, fd_mean, fd_std, conv_inv_fd = partial_generation(diffusion)
        # f.write('#Partial Generation\n')
        # f.write(str(apd_mean) + ',' + str(apd_std) + ',' + str(conv_inv_apd) + ',' +str(nn_mean) + ',' + str(nn_std)+ ',' + str(conv_inv_nn) + ',' + str(fd_mean) + ',' +str(fd_std)+ ',' + str(conv_inv_fd) + "\n")
        geo_m2m_mean, geo_m2m_std, m2m_dist_mean, m2m_dist_std = denoising_generation(diffusion)
        f.write('#Denoising Generation\n')
        f.write(str(geo_m2m_mean) + ',' + str(geo_m2m_std) + ',' + str(m2m_dist_mean) + ',' +str(m2m_dist_std))


