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

from Renderer import Renderer

def images_to_grid(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    im = torchvision.transforms.ToPILImage()(grid) 
    # im = ImageOps.invert(im)
    im.save(path)

def render_obj(render, verts, faces, save_loc):
    obj = render.save_obj(verts, faces, save_loc)

def render_img(render, verts, faces):
    img = render(verts, faces)
    # img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def visualise(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bdata = np.load(args['directory'] + args['frame'])
    bm_fname = osp.join(args['directory'], args['model'])

    betas = np.array([ 0.74854454, -0.6544366 , -0.07826481, -1.7522459 , -0.50113552,
        1.98938656, -1.56657068,  1.70431944, -2.41511792, -0.36359406,
       -0.8305809 , -2.36272325, -0.70008147,  0.24171983,  2.43522097,
        0.30899699])
    num_betas = betas.shape[0] # number of body parameters

    bm = BodyModel(bm_fname, num_betas=num_betas, model_type='smplh').to(device)
    faces = torch.Tensor(c2c(bm.f))

    time_length = 2
    body_parms = {
        'pose_body': torch.Tensor(bdata['pose_body'][:time_length].reshape(time_length, -1)).to(device), # controls the body
        'betas': torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device), # controls the body shape. Body shape is static
    }

    if args['print']: print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    if args['print']: print('time_length = {}'.format(time_length))
  
    body_pose_params = bm(pose_body=body_parms['pose_body'])

    imw, imh=800, 800
    renerer = Renderer(imw, imh)

    print('Saving images')

    if args['save_grid']:
        images = []
        
        for i in range(time_length):
            verts = (body_pose_params.v[i] - torch.mean(body_pose_params.v[i], dim = 1, keepdim=True)).cpu()
            img = render_img(renerer, verts, faces)
            if args['output_obj']:
                render_obj(renerer, verts, faces, args['directory'] + args['image_loc'] + "{:03d}.obj".format(i))

            images.append(img.permute(2,0,1))
        
        images_to_grid(images, args['directory'] + args['image_loc'] + args['name'] + "grid.png", nrow=4)

    else:

        for i in range(time_length):
            verts = (body_pose_params.v[i] - torch.mean(body_pose_params.v[i], dim = 1, keepdim=True)).cpu()
            img = render_img(renerer, verts, faces)
            if args['output_obj']:
                render_obj(renerer, verts, faces, args['directory'] + args['image_loc'] + "{:03d}.obj".format(i))

            plt.imsave(args['directory'] + args['image_loc'] + "{:03d}.png".format(i), img.numpy())
            print(f"saved image {i}")




if __name__ == '__main__':
    args = {
        # 'support_dir': '/vol/bitbucket/mew23/individual_project/',
        'directory': '/vol/bitbucket/mew23/individual-project/',
        'frame': 'samples/gen_video/data_0.npz',
        'model': './dataset/models/neutral/model.npz',
        'image_loc': 'samples/gen_video/images/',
        'name': '',
        'print': True,

        'output_obj': True,

        'save_grid': False,
        

    }

    visualise(args)
