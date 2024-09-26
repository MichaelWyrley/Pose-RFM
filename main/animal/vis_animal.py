import sys
sys.path.append('')
# Taken from https://github.com/nghorbani/amass/blob/a9888a92a4e62533454aa43e5f979d9a8bc8c893/notebooks/01-AMASS_Visualization.ipynb
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ["EGL_DEVICE_ID"] = "1"
# os.environ['DISPLAY'] = ':0.0'
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from PIL import Image
from PIL import ImageOps

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp
from main.utils.image.visualise_torch3d import images_to_grid
from main.utils.image.Renderer import Renderer
import pickle as pkl


# This code is directly copied from SMALViewer (https://github.com/benjiebob/SMALViewer) and flame-fitting (https://github.com/Rubikplayer/flame-fitting)
# It hasn't been added directly to the repository as animal generation is not the main aim
from main.animal.smal_model.smal_model import SMAL
# from notmycode.Renderer import Renderer

# label_to_betas = [20, 5, 38, 33, 31]
label_to_betas = [22, 0, 38, 33, 37]

def initialise_body(body_model_file, body_data_file, sym_file):
    smal_params = {
        'betas' : torch.zeros(1, 41),
        'joint_rotations' : torch.zeros(1, 34, 3),
        'global_rotation' :  torch.zeros(1, 1, 3),
        'trans' : torch.zeros(1, 1, 3),
    }
    smal_params['global_rotation'][:,:,0] = -np.pi / 2

    smal_model = SMAL(body_model_file, body_data_file, sym_file=sym_file)

    with open(body_data_file, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        smal_data = u.load()

    animal_betas = torch.from_numpy(smal_data['toys_betas']).to(torch.float32)

    return smal_model, animal_betas, smal_params

def visualise(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bdata = np.load(args['frame'])
    time_length = min(args['time_length'], bdata['pose_body'].shape[0])

    imw, imh=800, 800
    renerer = Renderer(imw, imh)
    
    pose_body = torch.from_numpy(bdata['pose_body']).to(torch.float32) # controls the body
    categories = bdata['categories'][:time_length]

    smal_model, animal_betas, smal_params = initialise_body(args['model'], args['model_data'], args['sym_file'])

    images = []
    for i in range(time_length):
        print(f"generating image: {i}")
        smal_params['betas'] = animal_betas[label_to_betas[categories[i]]].unsqueeze(0)
        smal_params['joint_rotations'] = pose_body[i].unsqueeze(0)

        verts, joints, Rs, v_shaped = smal_model(
            smal_params['betas'], 
            torch.cat([smal_params['global_rotation'], smal_params['joint_rotations']], dim=1))

        # normalize by center of mass
        verts = verts - torch.mean(verts, dim = 1, keepdim=True)
        img = renerer(verts, smal_model.faces.unsqueeze(0))
        if args['output_obj']:
                renerer.save_obj(verts, smal_model.faces.unsqueeze(0), args['image_loc'] + "{:03d}.obj".format(i))
    
        images.append(img.permute(2,0,1))

    images_to_grid(images, args['image_loc'] + args['name'] + "grid.pdf", nrow=5)




if __name__ == '__main__':
    args = {
        'frame': 'dataset/animal3d/SAMPLED_POSES/test.npz',
        'model': 'dataset/animal3d/MODELS/smpl_models/my_smpl_00781_4_all.pkl',
        'model_data': 'dataset/animal3d/MODELS/smpl_models/my_smpl_data_00781_4_all.pkl',
        'sym_file': 'dataset/animal3d/MODELS/smpl_models/symIdx.pkl',
        'image_loc': 'samples/animal_images/',
        'time_length': 1,
        'name': '',
        'output_obj': False,
    }

    # os.makedirs(args['image_loc'], exist_ok=True)

    visualise(args)
    # animal_types(args)
