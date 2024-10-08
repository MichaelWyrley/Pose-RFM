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
import cv2
import torchvision
from PIL import Image
from PIL import ImageOps

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp
from human_body_prior.body_model.body_model import BodyModel

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

def images_to_grid(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    im = torchvision.transforms.ToPILImage()(grid) 
    im = ImageOps.invert(im)
    im.save(path)

def vis_body_pose_beta(body_pose_beta, faces, mv, fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    return body_image


def visualise(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args['print']: print(device)

    bdata = np.load(args['frame'])

    if args['print']: print('Data keys available:%s'%list(bdata.keys()))

    num_betas = 10 # number of body parameters

    bm = BodyModel(args['model'], num_betas=num_betas, model_type='smplh').to(device)
    faces = c2c(bm.f)

    time_length = min(bdata['pose_body'].shape[0], args['time_length'])
    # time_length = 16
    
    print(bdata['pose_body'].shape)
    if 'betas' in bdata.keys():
        body_parms = {
            'pose_body': torch.Tensor(bdata['pose_body'][:time_length].reshape(time_length, -1)).to(device), # controls the body
            'betas': torch.Tensor(np.repeat(bdata['betas'][np.newaxis], repeats=time_length, axis=0)[:, :num_betas]).to(device), # controls the body shape. Body shape is static
        }  
        body_pose_beta = bm(pose_body=body_parms['pose_body'], betas = body_parms['betas'])
    else:
        body_parms = {
            'pose_body': torch.Tensor(bdata['pose_body'][:time_length].reshape(time_length, -1)).to(device), # controls the body
        }  
        body_pose_beta = bm(pose_body=body_parms['pose_body'])

    if args['print']: print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    if args['print']: print('time_length = {}'.format(time_length))

    imw, imh=800, 800
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    if args['save_grid']:
        images = []
        for i in range(time_length):
            img = vis_body_pose_beta(body_pose_beta, faces, mv, fId=i)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(torch.Tensor(img).permute(2,0,1))
        
        images_to_grid(images, args['image_loc'] + args['name'] + "grid.png", nrow=4)

    else:

        for i in range(time_length):
            img = vis_body_pose_beta(body_pose_beta, faces, mv, fId=i)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imsave(args['image_loc'] + "{:03d}.png".format(i), img)
            print(f"saved image {i}")




if __name__ == '__main__':
    args = {
        # 'frame': 'dataset/3DPW/smpl_poses/downtown_stairs_00.npz',
        'frame': 'dataset/3DPW/npz_poses/ground_truth/downtown_stairs_00.npz',
        'model': './dataset/models/neutral/model.npz',
        'image_loc': './samples/images/',
        'name': '',
        'print': True,
        'time_length': 2,

        'save_grid': False,
    }

    # os.mkdir(args['image_loc'], exist_ok=True)

    visualise(args)
