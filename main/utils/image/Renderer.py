# modified from https://github.com/benjiebob/SMALViewer/blob/master/p3d_renderer.py
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    PointLights, SoftPhongShader, SoftSilhouetteShader, TexturesVertex
)
from pytorch3d.io import load_objs_as_meshes, save_obj, IO

import torch
import torch.nn as nn
import numpy as np

import cv2

class Renderer(nn.Module):
    def __init__(self, img_width, img_height):
        super(Renderer, self).__init__()

        self.image_size = (img_width, img_height)
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=None
        )

        R, T = look_at_view_transform(2.7, 0, 0) 
        cameras = OpenGLPerspectiveCameras(device=R.device, R=R, T=T)
        lights = PointLights(device=R.device, location=[[2.0, 2.0, 0.0]])
        # blend_params = BlendParams(device=R.device, background_color = (0.0,0.0,0.0))

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=R.device, 
                cameras=cameras,
                lights=lights,
                # blend_params=blend_params
            )
        )

    def forward(self, verts, faces):
        verts_rgb = torch.ones_like(verts)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(verts.device))

        mesh = Meshes(verts=verts, faces=faces, textures = textures)
        img = self.renderer(mesh)[:, :,:,:3]
        # img = img.astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.reshape(self.image_size[0], self.image_size[1], -1)

    def save_obj(self, verts, faces, save_loc):
        verts_rgb = torch.ones_like(verts)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(verts.device))

        mesh = Meshes(verts=verts, faces=faces, textures = textures)
        IO().save_mesh(mesh, save_loc, include_textures = True)

