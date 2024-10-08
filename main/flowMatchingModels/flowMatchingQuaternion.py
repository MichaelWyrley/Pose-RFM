# Currently Not Used 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from main.flowMatchingModels.flowMatching import FlowMatching
from main.utils.NRDF.utils.data_utils import log_map, exp_map, cal_intrinsic_geo
from main.utils.NRDF.utils.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle

import pytorch3d.transforms as transforms

class FlowMatchingQuaternion(FlowMatching):
    def __init__(self, model, device = 'cuda'):
        """
        :param model: The predictive model used for vector field predictions
        :param device: The computational device (default: 'cuda')
        """
        self.device = device
        self.model = model

    def sample_timestep(self, n):
        """
        Generates a uniformly random timestep for each of the n samples.
        
        :param n: Number of timestep samples to generate.
        :return: A tensor of shape (n, 1) containing timesteps, placed on the specified device.
        """
        # u_0 = torch.rand(1)
        # t = torch.tensor([(u_0 + i / n) % 1 for i in range(n) ])
        # return t.unsqueeze(-1).to(self.device)
        return torch.rand(n, device=self.device).unsqueeze(-1)

    def gen_random_x(self, x_1):
        """
        Generates a random quaternion for each element in x_1.
        
        :param x_1: The tensor of shape [batch size, bones, quaternion]
        :return: A tensor of random quaternions matching the shape of x_1.
        """
        size = x_1.shape[0] * x_1.shape[1]
        x = transforms.random_quaternions(size, device = self.device)
        x = x.view(x_1.shape)
        
        return x
        
    def apply_nn(self, x, t):
        """
        Apply the neural network model to rotations to predict the conditional vector field.
        
        :param x: Input quaternion
        :param t: Time parameter
        :return: Predicted changes in the state as rotation matrices
        """
        v_t = self.model(x, t)

        # v_t = v_t / torch.linalg.norm(v_t, dim=2)
        
        return v_t

    # Modified FROM https://github.com/hynann/NRDF/blob/master/lib/core/rdfgrad.py
    def egrad2rgrad(self, egrad, q, norm=True):
        """
        the projection and mapping onto the tangent space of the canonical unit quaternion, 
        preserving the constraints of quaternions.
        
        :param egrad: Euclidean gradient tensor.
        :param q: Quaternion tensor associated with the gradient.
        :param norm: Whether to normalize the output.
        :return: The corresponding Riemannian gradient.
        """
        bs, n_joints, _ = q.shape

        Id = torch.eye(4).to(self.device)
        Id = Id.expand(bs, n_joints, 4, 4)
        P = Id - torch.einsum('...ij, ...jk -> ...ik', q.unsqueeze(-1), q.unsqueeze(2)) # (bs, nj, 4, 4)
        
        # project egrad to the tangent of q -> v
        v = torch.einsum('...ij, ...jk -> ...ik', P, egrad.unsqueeze(-1)) # (bs, nj, 4, 1)
        if norm:
            v = torch.nn.functional.normalize(v, dim=2) # unit gradient length

        # unit quaternion constraint
        rmat = torch.eye(4) # (bs, nj, 4, 4)
        rmat = rmat.expand(bs, n_joints, 4, 4).to(self.device)

        rmat[:, :, 0, 0] = 0.
        rmat[:, :, 1, 0] = -q[:, :, 1] / (1 + q[:, :, 0])
        rmat[:, :, 2, 0] = -q[:, :, 2] / (1 + q[:, :, 0])
        rmat[:, :, 3, 0] = -q[:, :, 3] / (1 + q[:, :, 0])
        
        rgrad = torch.einsum('...ij, ...jk -> ...ik', rmat, v)
        rgrad = rgrad.squeeze(-1) # [bs, nj, 4]

        return rgrad

    def conditional_flow(self, x_0, x_1, t):
        """
        Compute the conditional flow using Riemanian Flow Matching from paper "Flow Matching on General Geometries"
        
        :param x_0: Initial state rotation matrices
        :param x_1: Final state rotation matrices
        :param t: Time parameter for flow computation
        :return: The conditional flow as rotation matrices
        """
        log_x0_x1 = log_map(x_0, x_1)
        out = exp_map(x_0, t * log_x0_x1)
  
        return out
    
    def conditional_vector_field(self, x_0, x_t, x_1, t, epsilon=0.00001):
        """
        Compute the conditional vector field that minimizes the distance in the tangent space of the Lie group.
        
        :param x_0: Initial state rotation matrices
        :param x_t: Intermediate state rotation matrices
        :param x_1: Final state rotation matrices
        :param t: Time parameter for vector field computation
        :param epsilon: Small value to prevent division by zero
        :return: The conditional vector field
        """
        d_0_1 = cal_intrinsic_geo(x_0, x_1)
        d_t_1 = cal_intrinsic_geo(x_t, x_1)

        grad_d_t_1 = torch.autograd.grad(
            inputs=x_t,
            outputs=d_t_1,
            grad_outputs=torch.ones_like(d_0_1),
            create_graph=True,
            retain_graph=True)[0]

        
        out = d_0_1.unsqueeze(-1) * grad_d_t_1 / ((torch.linalg.norm(grad_d_t_1, dim=2)+epsilon).unsqueeze(-1))
        
        if (out.isnan().any()):
            out = torch.nan_to_num(out, nan=0.0)

        return out

    def train_step(self, x_1):
        """
        Perform a training step using input data x_1 to optimize the predictive model.
        
        :param x_1: Input data used for training
        :return: The computed loss as a result of training
        """
        x_1 = axis_angle_to_quaternion(x_1)
        x_0 = self.gen_random_x(x_1)
        t = self.sample_timestep(x_0.shape[0]).requires_grad_(True)

        psi_t = self.conditional_flow(x_0, x_1, t[:, None])
        v_t = self.apply_nn(psi_t, t)

        con_vec = self.conditional_vector_field(x_0, psi_t, x_1, t)

        loss = F.mse_loss(v_t, -con_vec)

        return loss

    def sample(self, n, timesteps=100):
        """
        Generate a sample path using the model's dynamics over specified timesteps.
        
        :param n: Number of samples to generate
        :param timesteps: Number of timesteps to simulate
        :param scale: Scaling factor for timestep computations
        :return: Final state after simulation
        """

        self.model.eval()
        with torch.no_grad():
            z = self.gen_random_x(torch.zeros(n, 21, 4)).to(self.device)
            steps = torch.linspace(0.0, 1.0, timesteps, device=self.device)
            for i in range(timesteps):
                t = torch.full((n,1), steps[i], device=self.device)
                v_t = self.model(z, t)
                # z = z + 1 / timesteps * v_t
                v_t = self.egrad2rgrad(v_t, z)
                z = exp_map(z, -(1/timesteps) * v_t)
                
        self.model.train()

        return quaternion_to_axis_angle(z)

    def sample_full(self, n, timesteps=50):
        """
        Generate a sample path using the model's dynamics over specified timesteps.
        
        :param n: Number of samples to generate
        :param timesteps: Number of timesteps to simulate
        :param scale: Scaling factor for timestep computations
        :return: Full path taken for each sample
        """

        zs = []
        self.model.eval()
        with torch.no_grad():
            z = self.gen_random_x(torch.zeros(n, 21, 4)).to(self.device)
            steps = torch.linspace(0.0, 1.0, timesteps, device=self.device)
            for i in range(timesteps):
                zs.append(quaternion_to_axis_angle(z))
                t = torch.full((n,1), steps[i], device=self.device)
                v_t = self.model(z, t)
                # z = z + 1 / timesteps * v_t
                v_t = self.egrad2rgrad(v_t, z)
                z = exp_map(z, -(1/timesteps) * v_t)
                
        self.model.train()
        zs.append(quaternion_to_axis_angle(z))
        return quaternion_to_axis_angle(z), torch.stack(zs)


