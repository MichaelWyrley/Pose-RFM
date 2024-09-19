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

def vis_body_joints(body_pose_hand, mv, fId = 0):
    joints = c2c(body_pose_hand.Jtr[fId])
    joints_mesh = points_to_spheres(joints, point_color = colors['red'], radius=0.005)

    mv.set_static_meshes([joints_mesh])
    body_image = mv.render(render_wireframe=False)
    return body_image

def vis_body_pose_beta(body_pose_beta, faces, mv, fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    return body_image

def vis_body_pose_hand(body_pose_hand, faces, mv, fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    return body_image


def visualise(args):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args['print']: print(device)

    bdata = np.load(args['directory'] + args['frame'])

    # print(bdata['pose_body'].shape)

    if args['print']: print('Data keys available:%s'%list(bdata.keys()))

    bm_fname = osp.join(args['directory'], args['model'])

    betas = np.array([ 0.74854454, -0.6544366 , -0.07826481, -1.7522459 , -0.50113552,
        1.98938656, -1.56657068,  1.70431944, -2.41511792, -0.36359406,
       -0.8305809 , -2.36272325, -0.70008147,  0.24171983,  2.43522097,
        0.30899699])
    # dmpl_fname = osp.join(args['directory'], 'body_models/dmpls/{}/model.npz'.format(subject_gender))

    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    bm = BodyModel(bm_fname, num_betas=num_betas, model_type='smplh').to(device)
    faces = c2c(bm.f)

    time_length = len(bdata['pose_body'])
    # time_length = 16
    
    # time_length = len(bdata['trans'])

    body_parms = {
        # 'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['pose_body'][:time_length].reshape(time_length, -1)).to(device), # controls the body
        # 'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(device), # controls the finger articulation
        # 'trans': torch.Tensor(bdata['trans']).to(device), # controls the global body position
        'betas': torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device), # controls the body shape. Body shape is static
        # 'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(device) # controls soft tissue dynamics
    }

    if args['print']: print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    if args['print']: print('time_length = {}'.format(time_length))
  
    body_pose_beta = bm(pose_body=body_parms['pose_body'])

    imw, imh=800, 800
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    if args['save_grid']:
        images = []
        for i in range(time_length):
            img = vis_body_pose_beta(body_pose_beta, faces, mv, fId=i)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(torch.Tensor(img).permute(2,0,1))
        
        images_to_grid(images, args['directory'] + args['image_loc'] + args['name'] + "grid.png", nrow=4)

    else:

        for i in range(time_length):
            img = vis_body_pose_beta(body_pose_beta, faces, mv, fId=i)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imsave(args['directory'] + args['image_loc'] + "{:03d}.png".format(i), img)
            print(f"saved image {i}")




if __name__ == '__main__':
    args = {
        # 'support_dir': '/vol/bitbucket/mew23/individual_project/',
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'frame': 'samples/gen_video/data_0.npz',
        'model': './dataset/models/neutral/model.npz',
        'image_loc': './samples/images/',
        'name': '',
        'print': True,

        'save_grid': True,
        

    }

    visualise(args)
