import sys
sys.path.append('')
# modified from https://github.com/nghorbani/amass/blob/a9888a92a4e62533454aa43e5f979d9a8bc8c893/notebooks/01-AMASS_Visualization.ipynb
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ["EGL_DEVICE_ID"] = "1"
# os.environ['DISPLAY'] = ':0.0'
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp
from human_body_prior.body_model.body_model import BodyModel

from main.utils.image.Renderer import Renderer

def images_to_grid(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    im = torchvision.transforms.ToPILImage()(grid) 
    # im = ImageOps.invert(im)
    im.save(path)

def render_obj(render, verts, faces, save_loc):
    render.save_obj(verts, faces, save_loc)

def render_img(render, verts, faces):
    img = render(verts, faces)
    return img

def visualise(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bdata = np.load(args['frame'])
    bm_fname = osp.join(args['model'])

    num_betas = 10 # number of body parameters

    bm = BodyModel(args['model'], num_betas=num_betas, model_type='smplh').to(device)
    faces = torch.Tensor(c2c(bm.f))

    time_length = 2
    print(bdata['poses_body'].shape)
    if 'betas' in bdata.keys():
        body_parms = {
            'pose_body': torch.Tensor(bdata['poses_body'][0, :time_length].reshape(time_length, -1)[:, 3:66]).to(device), # controls the body
            'betas': torch.Tensor(np.repeat(bdata['betas'][0, np.newaxis], repeats=time_length, axis=0)[:, :10]).to(device), # controls the body shape. Body shape is static
        }  
        body_pose_params = bm(pose_body=body_parms['pose_body'], betas = body_parms['betas'])
    else:
        body_parms = {
            'pose_body': torch.Tensor(bdata['poses_body'][0, :time_length].reshape(time_length, -1)[:, 3:66]).to(device), # controls the body
        }  
        body_pose_params = bm(pose_body=body_parms['pose_body'])

    if args['print']: print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    if args['print']: print('time_length = {}'.format(time_length))

    imw, imh=800, 800
    renerer = Renderer(imw, imh)

    print('Saving images')

    if args['save_grid']:
        images = []
        
        for i in range(time_length):
            verts = (body_pose_params.v[i] - torch.mean(body_pose_params.v[i], dim = 1, keepdim=True)).cpu()
            img = render_img(renerer, verts, faces)
            if args['output_obj']:
                render_obj(renerer, verts, faces, args['image_loc'] + "{:03d}.obj".format(i))

            images.append(img.permute(2,0,1))
        
        images_to_grid(images, args['image_loc'] + args['name'] + "grid.png", nrow=4)

    else:

        for i in range(time_length):
            verts = (body_pose_params.v[i] - torch.mean(body_pose_params.v[i], dim = 1, keepdim=True)).cpu()
            img = render_img(renerer, verts, faces)
            if args['output_obj']:
                render_obj(renerer, verts, faces, args['image_loc'] + "{:03d}.obj".format(i))

            plt.imsave(args['image_loc'] + "{:03d}.png".format(i), img.numpy())
            print(f"saved image {i}")




if __name__ == '__main__':
    args = {
        'frame': 'samples/gen_video/data_0.npz',
        'model': './dataset/models/neutral/model.npz',
        'image_loc': 'samples/gen_video/images/',
        'name': '',
        'print': True,

        'output_obj': True,

        'save_grid': False,
    }

    os.mkdir(args['image_loc'], exist_ok=True)

    visualise(args)
