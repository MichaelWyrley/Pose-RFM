import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def test_different_models(file, directory='/vol/bitbucket/mew23/individual-project/', name = 'test_different_models.png'):
    data = {
        'epoch': [],
        'APD_mu': [],
        'APD_std': [],
        'FID_mu': [],
        'FID_std': [],
    }

    with open(directory+file, 'r') as f:
        for line in f:
            if 'MODEL' in line:
                epoch = line.split('/')[-1].split('.')[0].split('_')[1]
                data['epoch'].append(epoch)
            else:
                apd_mu, apd_std, fid_mu, fid_std = line.split(',')
                data['APD_mu'].append(float(apd_mu))
                data['APD_std'].append(float(apd_std))
                data['FID_mu'].append(float(fid_mu))
                data['FID_std'].append(float(fid_std))
    
    fig, ax = plt.subplots(2, figsize=(20,10))
    fig.tight_layout(pad=3.0)
    ax[0].set(xlabel='Epoch', ylabel='APD (cm)')
    ax[1].set(xlabel='Epoch', ylabel='FID')

    ax[0].errorbar(data['epoch'], data['APD_mu'], yerr=data['APD_std'], capsize=3, ecolor = "black")
    ax[1].errorbar(data['epoch'], data['FID_mu'], yerr=data['FID_std'], capsize=3, ecolor = "black")

    fig.savefig(directory + 'experiments/graphs/' + name)

def test_normal_vs_ema_models(ema_file, normal_file, directory='/vol/bitbucket/mew23/individual-project/', name = 'test_different_models_1200.png'):
    data_ema = {
        'epoch': [],
        'APD_mu': [],
        'APD_std': [],
        'FID_mu': [],
        'FID_std': [],
    }

    data = {
        'epoch': [],
        'APD_mu': [],
        'APD_std': [],
        'FID_mu': [],
        'FID_std': [],
    }

    with open(directory+ema_file, 'r') as f:
        for line in f:
            if 'MODEL' in line:
                epoch = line.split('/')[-1].split('.')[0].split('_')[2]
                data_ema['epoch'].append(epoch)
            else:
                apd_mu, apd_std, fid_mu, fid_std = line.split(',')
                data_ema['APD_mu'].append(float(apd_mu))
                data_ema['APD_std'].append(float(apd_std))
                data_ema['FID_mu'].append(float(fid_mu))
                data_ema['FID_std'].append(float(fid_std))

    with open(directory+normal_file, 'r') as f:
        for line in f:
            if 'MODEL' in line:
                epoch = line.split('/')[-1].split('.')[0].split('_')[1]
                data['epoch'].append(epoch)
            else:
                apd_mu, apd_std, fid_mu, fid_std = line.split(',')
                data['APD_mu'].append(float(apd_mu))
                data['APD_std'].append(float(apd_std))
                data['FID_mu'].append(float(fid_mu))
                data['FID_std'].append(float(fid_std))
    
    fig, ax = plt.subplots(2, figsize=(20,10))
    fig.tight_layout(pad=3.0)
    ax[0].set(xlabel='Epoch', ylabel='APD (m)')
    ax[1].set(xlabel='Epoch', ylabel='FID')

    ax[0].errorbar(data_ema['epoch'], data_ema['APD_mu'], yerr=data_ema['APD_std'], capsize=3, ecolor = "black", label='EMA Model')
    ax[1].errorbar(data_ema['epoch'], data_ema['FID_mu'], yerr=data_ema['FID_std'], capsize=3, ecolor = "black")

    ax[0].errorbar(data['epoch'], data['APD_mu'], yerr=data['APD_std'], capsize=3, ecolor = "black", label='Model')
    ax[1].errorbar(data['epoch'], data['FID_mu'], yerr=data['FID_std'], capsize=3, ecolor = "black")

    ax[0].legend(loc='upper right', ncols=6)

    fig.savefig(directory + 'experiments/graphs/' + name)

def test_all_different_models(file, directory='/vol/bitbucket/mew23/individual-project/', name = 'test_time_different_models.png'):
    data = {
        'epoch': [],
        'APD_mu': [],
        'APD_std': [],
        'FID_mu': [],
        'FID_std': [],
    }

    with open(directory+file, 'r') as f:
        for line in f:
            if 'MODEL' in line:
                epoch = line.split('/')[-2].split('_')[-1]
                data['epoch'].append(epoch + '%')
            else:
                apd_mu, apd_std, fid_mu, fid_std = line.split(',')
                data['APD_mu'].append(float(apd_mu))
                data['APD_std'].append(float(apd_std))
                data['FID_mu'].append(float(fid_mu))
                data['FID_std'].append(float(fid_std))
    
    fig, ax = plt.subplots(2, figsize=(20,10))
    fig.tight_layout(pad=3.0)
    ax[0].set(xlabel='Model', ylabel='∇q + v2v')
    ax[1].set(xlabel='Model', ylabel='v2v (m)')

    ax[0].bar(data['epoch'], data['APD_mu'], yerr=data['APD_std'], capsize=3, ecolor = "black")
    ax[1].bar(data['epoch'], data['FID_mu'], yerr=data['FID_std'], capsize=3, ecolor = "black")

    fig.savefig(directory + 'experiments/graphs/' + name)

def test_differnt_no_samples_and_scales(file, directory='/vol/bitbucket/mew23/individual-project/', name = 'test_differnt_no_samples_and_scales.png'):

    scales = []
    current_item = -1

    with open(directory+file, 'r') as f:
        for line in f:
            if '-----------------------------------------\n' == line:
                scales.append({
                    'timestep': [],
                    'APD_mu': [],
                    'APD_std': [],
                    'FID_mu': [],
                    'FID_std': [],
                    'scale': '',
                })
                current_item += 1

            elif 'timestep' in line:
                l = line.split(',')
                timestep, scale = l[0].split('=')[1], l[1].split('=')[1]
                scales[current_item]['timestep'].append(timestep)
                # scales[current_item]['scale'] = scale

            elif 'working_models' in line:
                epoch = line.split('/')[-2].split('_')[-1]
                scales[current_item]['scale'] = epoch 
            else:
                apd_mu, apd_std, fid_mu, fid_std = line.split(',')
                scales[current_item]['APD_mu'].append(float(apd_mu))
                scales[current_item]['APD_std'].append(float(apd_std))
                scales[current_item]['FID_mu'].append(float(fid_mu))
                scales[current_item]['FID_std'].append(float(fid_std))
    
    fig, ax = plt.subplots(2, figsize=(25,15))
    fig.tight_layout(pad=3.0)


    for scale in scales: 
        ax[0].errorbar(scale['timestep'], scale['APD_mu'], yerr=scale['APD_std'], capsize=3, ecolor = "black", label=scale['scale'])

        ax[1].errorbar(scale['timestep'], scale['FID_mu'], yerr=scale['FID_std'], capsize=3, ecolor = "black", label=scale['scale'])

    ax[0].set(xlabel='Timestep', ylabel='APD (m)')
    ax[1].set(xlabel='Timestep', ylabel='FID')
    ax[0].legend(loc='upper left', ncols=6, bbox_to_anchor=(.55,1.15))
    
    # ax[1].legend(loc='upper left', ncols=6)

    fig.savefig(directory + 'experiments/graphs/' + name)


if __name__ == '__main__':
    # test_different_models('experiments/current_tests/test_different_models_ema_0-1200.txt', name = 'different_models_ema_0-1200.png')
    # test_normal_vs_ema_models('experiments/current_tests/test_different_models_ema_0-1200.txt', 'experiments/current_tests/test_different_models_0-1200.txt', name = 'different_models_ema_0-1200.png')
    # test_all_different_models('experiments/current_tests/test_different_models_noise_prob.txt', name = 'different_models_noise_prob.png')
    # test_all_different_models('experiments/current_tests/test_denoising_different_models.txt', name = 'test_denoising_different_models.png')
    test_differnt_no_samples_and_scales('experiments/current_tests/test_differnt_no_samples_and_scales_time.txt', name = 'different_scales_time.png')