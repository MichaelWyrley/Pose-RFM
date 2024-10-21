# modified from https://github.com/benjiebob/SMALViewer/blob/master/p3d_renderer.py
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    PointLights, SoftPhongShader, SoftSilhouetteShader, TexturesVertex
)
from pytorch3d.io import load_objs_as_meshes, save_obj, IO

import torch
import torch.nn as nn
import numpy as np

import cv2

class Renderer(nn.Module):
    def __init__(self, img_width=800, img_height=800, focal_length=5000, cam_intrinsics=None, device='cuda'):
        super(Renderer, self).__init__()
        self.device = device
        if cam_intrinsics is None:
            self.image_size = (img_width, img_height)
        else:
            self.image_size = (cam_intrinsics[0, 2]*2, cam_intrinsics[1, 2]*2)
            focal_length = torch.cat([cam_intrinsics[0, 0], cam_intrinsics[1, 1]]).unsqueeze(0)

        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=None
        )

        R, T = look_at_view_transform(2.7, 0, 0) 
        # cameras = PerspectiveCameras(focal_length = focal_length, device=device, R=R, T=T)
        cameras = OpenGLPerspectiveCameras(device=R.device, R=R, T=T)
        lights = PointLights(device=device, location=[[2.0, 2.0, 0.0]])
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
        ).to(device)

    def forward(self, verts, faces):
        verts_rgb = torch.ones_like(verts)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb).to(self.device)

        mesh = Meshes(verts=verts, faces=faces, textures = textures)
        img = self.renderer(mesh)[:, :,:,:3]
        # img = img.astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def save_obj(self, verts, faces, save_loc):
        verts_rgb = torch.ones_like(verts)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb).to(self.device)

        mesh = Meshes(verts=verts, faces=faces, textures = textures)
        IO().save_mesh(mesh, save_loc, include_textures = True)

