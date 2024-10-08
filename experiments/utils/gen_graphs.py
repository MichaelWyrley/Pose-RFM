import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.facecolor'] = '#fafafa'

bar_colours = ['#dae8fc', '#d5e8d4', '#e1d5e7', '#f8cecc', '#fff2cc', '#e8cae1']
line_colours = ['#6C8EBF', '#82B366', '#9673A6', '#B85450', '#D6B656', '#e0a8d3']


def test_normal_vs_ema_models(ema_file, normal_file, name = 'test_different_models_1200.png'):
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

    with open(ema_file, 'r') as f:
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

    with open(normal_file, 'r') as f:
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
    fig.tight_layout(pad=2.0)
    ax[0].set(xlabel='Epoch', ylabel='APD (m)')
    ax[1].set(xlabel='Epoch', ylabel='FD')

    ax[0].errorbar(data_ema['epoch'], data_ema['APD_mu'], yerr=data_ema['APD_std'], capsize=3, color=line_colours[0], ecolor = "black", label='EMA Model', linewidth=2.5)
    ax[1].errorbar(data_ema['epoch'], data_ema['FID_mu'], yerr=data_ema['FID_std'], capsize=3, color=line_colours[0], ecolor = "black", linewidth=2.5)

    ax[0].errorbar(data['epoch'], data['APD_mu'], yerr=data['APD_std'], capsize=3, color=line_colours[1],  ecolor = "black", label='Model', linewidth=2.5)
    ax[1].errorbar(data['epoch'], data['FID_mu'], yerr=data['FID_std'], capsize=3, color=line_colours[1], ecolor = "black", linewidth=2.5)

    ax[0].legend(loc='upper right', ncols=6)
    fig.savefig('experiments/graphs/' + name)

def test_all_different_models_noise_prob(file_apd_fid, file_v2v, name = 'different_models_noise_prob.png'):
    data_apd_fid = {
        'epoch': [],
        'APD_mu': [],
        'APD_std': [],
        'FID_mu': [],
        'FID_std': [],
    }
    data_v2v = {
        'epoch': [],
        'q_v2v_mu': [],
        'q_v2v_std': [],
        'v2v_mu': [],
        'v2v_std': [],
    }


    with open(file_apd_fid, 'r') as f:
        for line in f:
            if 'MODEL' in line:
                epoch = line.split('/')[-2].split('_')[-1]
                data_apd_fid['epoch'].append(epoch + '%')
            else:
                apd_mu, apd_std, fid_mu, fid_std = line.split(',')
                data_apd_fid['APD_mu'].append(float(apd_mu))
                data_apd_fid['APD_std'].append(float(apd_std))
                data_apd_fid['FID_mu'].append(float(fid_mu))
                data_apd_fid['FID_std'].append(float(fid_std))



    with open(file_v2v, 'r') as f:
        for line in f:
            if 'MODEL' in line:
                epoch = line.split('/')[-2].split('_')[-1]
                data_v2v['epoch'].append(epoch + '%')
            else:
                q_v2v_mu, q_v2v_std, v2v_mu, v2v_std = line.split(',')
                data_v2v['q_v2v_mu'].append(float(q_v2v_mu))
                data_v2v['q_v2v_std'].append(float(q_v2v_std))
                data_v2v['v2v_mu'].append(float(v2v_mu))
                data_v2v['v2v_std'].append(float(v2v_std))
    
    fig, ax = plt.subplots(2,2, figsize=(18,12))
    fig.tight_layout(pad=2.0)

    ax[0][0].set(xlabel='Model', ylabel='APD (m)')
    ax[0][1].set(xlabel='Model', ylabel='FD')
    ax[1][0].set(xlabel='Model', ylabel='∇q + v2v')
    ax[1][1].set(xlabel='Model', ylabel='v2v (m)')

    ax[0][0].bar(data_apd_fid['epoch'], data_apd_fid['APD_mu'], yerr=data_apd_fid['APD_std'], capsize=3, ecolor = "black", color=bar_colours, edgecolor=line_colours)
    ax[0][1].bar(data_apd_fid['epoch'], data_apd_fid['FID_mu'], yerr=data_apd_fid['FID_std'], capsize=3, ecolor = "black", color=bar_colours, edgecolor=line_colours)
    ax[1][0].bar(data_v2v['epoch'], data_v2v['q_v2v_mu'], yerr=data_v2v['q_v2v_std'], capsize=3, ecolor = "black", color=bar_colours, edgecolor=line_colours)
    ax[1][1].bar(data_v2v['epoch'], data_v2v['v2v_mu'], yerr=data_v2v['v2v_std'], capsize=3, ecolor = "black", color=bar_colours, edgecolor=line_colours)

    fig.savefig('experiments/graphs/' + name)


def test_differnt_no_samples_and_scales(file, name = 'test_differnt_no_samples_and_scales.png', bbox=(.53,1)):

    scales = []
    current_item = -1

    with open(file, 'r') as f:
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
                if scales[current_item]['scale'] == '':
                    scales[current_item]['scale'] = scale

            elif 'working_models' in line:
                epoch = line.split('/')[-2].split('_')[-1]
                scales[current_item]['scale'] = epoch 
            else:
                apd_mu, apd_std, fid_mu, fid_std = line.split(',')
                scales[current_item]['APD_mu'].append(float(apd_mu))
                scales[current_item]['APD_std'].append(float(apd_std))
                scales[current_item]['FID_mu'].append(float(fid_mu))
                scales[current_item]['FID_std'].append(float(fid_std))
    
    fig, ax = plt.subplots(2, figsize=(23,13.5))
    fig.tight_layout(pad=2.0)


    for i, scale in enumerate(scales): 
        ax[0].errorbar(scale['timestep'], scale['APD_mu'], yerr=scale['APD_std'], capsize=3, ecolor = "black", label=scale['scale'], color=line_colours[i], linewidth=2.5)
        ax[1].errorbar(scale['timestep'], scale['FID_mu'], yerr=scale['FID_std'], capsize=3, ecolor = "black", label=scale['scale'], color=line_colours[i], linewidth=2.5)

    ax[0].set(xlabel='Timestep', ylabel='APD (m)')
    ax[1].set(xlabel='Timestep', ylabel='FD')

    ax[0].legend(loc='upper left', ncols=6, bbox_to_anchor=bbox)
    
    # ax[1].legend(loc='upper left', ncols=6)

    fig.savefig('experiments/graphs/' + name)

def test_dit_vs_ada_zero(file, name = 'test_time_different_models.png'):
    data = {
        'epoch': [],
        'APD_mu': [],
        'APD_std': [],
        'FID_mu': [],
        'FID_std': [],
    }

    with open(file, 'r') as f:
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
    
    fig, ax = plt.subplots(1, 2, figsize=(14,4))
    fig.tight_layout(pad=2.0)
    ax[0].set(xlabel='Model', ylabel='APD (m)')
    ax[1].set(xlabel='Model', ylabel='FD')

    ax[0].bar(data['epoch'], data['APD_mu'], yerr=data['APD_std'], capsize=3, ecolor = "black", color=bar_colours, edgecolor=line_colours)
    ax[1].bar(data['epoch'], data['FID_mu'], yerr=data['FID_std'], capsize=3, ecolor = "black", color=bar_colours, edgecolor=line_colours)

    fig.savefig('experiments/graphs/' + name)

def test_time_different_models(file,name = 'different_models_noise_prob.png'):
    data = {
        'epoch': [],
        'APD_mu': [],
        'APD_std': [],
        'FID_mu': [],
        'FID_std': [],
    }


    with open(file, 'r') as f:
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

    
    fig, ax = plt.subplots(1,2, figsize=(18,6))
    fig.tight_layout(pad=2.0)

    ax[0].set(xlabel='Model', ylabel='APD (m)')
    ax[1].set(xlabel='Model', ylabel='FD')

    ax[0].bar(data['epoch'], data['APD_mu'], yerr=data['APD_std'], capsize=3, ecolor = "black", color=bar_colours, edgecolor=line_colours)
    ax[1].bar(data['epoch'], data['FID_mu'], yerr=data['FID_std'], capsize=3, ecolor = "black", color=bar_colours, edgecolor=line_colours)

    fig.savefig('experiments/graphs/' + name)


if __name__ == '__main__':
    # test_normal_vs_ema_models('experiments/current_tests/test_different_models_ema_0-1200.txt', 'experiments/current_tests/test_different_models_0-1200.txt', name = 'different_models_ema_0-1200.pdf')
    # test_all_different_models_noise_prob('experiments/current_tests/test_different_models_noise_prob.txt', 'experiments/current_tests/test_denoising_different_models.txt', name = 'different_models_noise_prob.pdf')
    test_differnt_no_samples_and_scales('experiments/current_tests/test_differnt_no_samples_and_scales_1200.txt', name = 'different_scales_1200.pdf', bbox=(.51,1.15))
    # test_differnt_no_samples_and_scales('experiments/current_tests/test_differnt_no_samples_and_scales_time.txt', name = 'different_scales_time.pdf', bbox=(.53,1))

    # test_dit_vs_ada_zero('experiments/current_tests/dit_vs_ada_zero.txt', name = 'dit_vs_ada_zero.pdf')
    # test_time_different_models('experiments/current_tests/time_different_models.txt', name = 'test_time_different_models.pdf')