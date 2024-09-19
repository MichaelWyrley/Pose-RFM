import torch
import os
import sys
sys.path.append('')
import matplotlib.pyplot as plt
import torch.nn as nn
import copy

from Transformer_adaLN_zero_conditioned import DiT_adaLN_zero
from FlowMatchingModels.flowMatchingMatrix import FlowMatchingMatrix

from dataloader import PoseData
from Utils.NRDF.utils.transforms import quaternion_to_axis_angle
import numpy as np
from Utils.Model.EMA import EMA

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

    model = DiT_adaLN_zero(in_dim=6, depth=16, emb_dimention=768, num_heads=16, num_classes=5).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    diffusion = FlowMatchingMatrix(model, device=rank, gen_x0=args['gen_x0'], time_prob=args['time_prob'])

    data_set = PoseData(mode='train', clean_dir=args['directory'] + args['clean'], batch_size=args['batch_size'], num_workers=6, stage=1, flip=False, random_poses=False)
    data_loader = data_set.get_loader()

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    print("Starting Training")
    for epoch in range(0, args['epochs']+1):
        for i, (clean_poses, categorie) in enumerate(data_loader):
            optimizer.zero_grad()
            clean_pose = clean_poses.to(rank)
            categorie = categorie.to(rank)
            noisy_pose = None

            loss = diffusion.train_step(clean_pose, noisy_pose, categorie)

            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            if i % 50 == 0:
                print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item()}")

        print(f"Epoch [{epoch}] Loss: {loss.item()}")

        if epoch % 50 == 0 and rank==0:
            torch.save(model.module.state_dict(), f"{args['directory']}./models/model_{epoch}.pt")
            torch.save(ema_model.module.state_dict(), f"{args['directory']}./models/ema_model_{epoch}.pt")


    destroy_process_group()

# rank: unique identifier for each process
# world_size: Total number of processes
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

if __name__ == '__main__':
    args = {
        # 'directory': '/vol/bitbucket/mew23/individual_project_diffusion_example/',
        'directory': './',

        'clean': 'dataset/animal3d/SAMPLED_POSES/',

        'gen_x0': 0.0,
        'time_prob': 0.25,

        'batch_size': 200,
        'lr': 1e-4,
        'epochs': 800,

        'world_size': torch.cuda.device_count(),
    }

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # train_single(args)
    train_multigpu(args)

