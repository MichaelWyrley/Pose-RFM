import sys
sys.path.append('')
import torch
import os
import copy
import numpy as np

from main.vectorFieldModels.Transformer_adaLN_zero import DiT_adaLN_zero
from main.flowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix
from main.utils.NRDF.data.dataloaders import Pose_Noise_Data
from main.utils.EMA import EMA

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Set up multiple GPU training instances, (based off how many GPUs you have available)
def train_multigpu(args):
    mp.spawn(train_multigpu_loop, args=(args['world_size'], args), nprocs=args['world_size'])
    
# Train each multi-GPU instance
# rank: GPU instance (device)
# world_size: number of GPUs available
# args: all other arguments (specified in __main__)
def train_multigpu_loop(rank, world_size, args):
    ddp_setup(rank, world_size)

    model = DiT_adaLN_zero(in_dim=6).to(rank)
    start_epoch = 0
    if args['load_model'] is not None:
        model.load_state_dict(torch.load(args['load_model']))
        model.train()
        start_epoch = args['start_epoch']
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    diffusion = FlowMatchingMatrix(model, device=rank, gen_x0=args['gen_x0'], time_prob=args['time_prob'])

    data_set = Pose_Noise_Data(mode='train', clean_dir=args['clean'], noisy_dir=args['noisy'], batch_size=args['batch_size'], num_workers=6, num_pts=args['no_samples'], stage=1, flip=False, random_poses=False)
    data_loader = data_set.get_loader()

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    print("Starting Training")
    for epoch in range(start_epoch, args['epochs']+1):
        for i, (noisy_pose, clean_poses) in enumerate(data_loader):
            optimizer.zero_grad()
            clean_pose = clean_poses.to(rank).reshape(-1, 21, 3)
            noisy_pose = noisy_pose.to(rank).reshape(-1, 21, 3)

            loss = diffusion.train_step(clean_pose, noisy_pose)

            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            if i % 50 == 0:
                print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item()}")

        print(f"Epoch [{epoch}] Loss: {loss.item()}")

        if epoch % 50 == 0 and rank==0:
            torch.save(model.module.state_dict(), f"./samples/training_models/model_{epoch}.pt")
            torch.save(ema_model.module.state_dict(), f"./samples/training_models/ema_model_{epoch}.pt")
        if epoch % 25 == 0 and rank==0:
            sampled = diffusion.sample(args['samples'], scale=8)
            pose_body = sampled.cpu().numpy()

            np.savez(f'./samples/training_samples/epoch_{epoch}.npz', pose_body=pose_body)

    torch.save(model.module.state_dict(), f"./samples/training_models/model_final.pt")
    torch.save(ema_model.module.state_dict(), f"./samples/training_models/ema_model_final.pt")

    sampled = diffusion.sample(args['samples'], scale=8)
    pose_body = sampled.cpu().numpy()

    np.savez(f'./samples/training_samples/sample_final.npz', pose_body=pose_body)

    destroy_process_group()

# rank: unique identifier for each process
# world_size: Total number of processes
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

if __name__ == '__main__':

    args = {
        'noisy': 'dataset/amass/NOISY_POSES/gaussian_0.785',
        'clean': 'dataset/amass/SAMPLED_POSES',

        'load_model': None,
        'start_epoch': 0,

        'gen_x0': 0.9,
        'time_prob': 0.25,

        'batch_size': 2,
        'no_samples': 225,
        'lr': 1e-4,
        'epochs': 0,
        'samples': 16,

        'world_size': torch.cuda.device_count(),
    }
    
    # os.mkdir('./samples', exist_ok=True)
    # os.mkdir('./samples/training_models', exist_ok=True)
    # os.mkdir('./samples/training_samples', exist_ok=True)

    train_multigpu(args)