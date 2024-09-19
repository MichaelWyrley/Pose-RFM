import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.transforms as transforms

from Utils.NRDF.utils.transforms import axis_angle_to_quaternion
from FlowMatchingModels.flowMatching import FlowMatching

class FlowMatchingMatrix(FlowMatching):
    def __init__(self, model, device = 'cuda'):
        """
        :param model: The predictive model used for vector field predictions
        :param device: The computational device (default: 'cuda')
        """

        self.device = device
        self.model = model

    def sample_timestep(self, n):
        """
        Sample a single timestep for n samples, distributing the timesteps uniformly or randomly based on a random threshold.
        
        :param n: Number of timestep samples to generate
        :return: Tensor of sampled timesteps, shaped for batch processing and moved to the specified device
        """

        u_0 = torch.rand(1)

        if u_0 < 0.25:
            t = torch.tensor([(u_0 + i / n) % 1 for i in range(n) ])
            return t.unsqueeze(-1).to(self.device)
        else:
            return torch.rand(n, device=self.device).unsqueeze(-1)

    def gen_random_x(self, x_1):
        """
        Generate a random rotation matrix (SO(3)) based on the size and shape of x_1.
        
        :param x_1: The tensor of shape [batch size, bones, rotation matrix part, rotation matrix part]
        :return: A randomly rotated matrix reshaped to the shape of x_1 
        """
        size = x_1.shape[0] * x_1.shape[1] * x_1.shape[2]
        x = transforms.random_rotations(size, device = self.device)
        x = x.view(x_1.shape)
        
        return x

    def exp_map(self, a, b):
        """
        Exponential map for SO(3) applied between two rotation matrixes ().
        
        :param a: Initial rotation matrices
        :param b: Lie algebra elements of Skew-symmetric matrices
        :return: Resultant rotation matrices after applying the exponential map
        """

        batch, f, n, _ = b.shape
        b = b.view(-1, 3)

        out = transforms.so3_exponential_map(b).view(batch, f, n, 3, 3)

        return out @ a

    def log_map(self, a, b):
        """
        Logarithm map for SO(3) used to compute Lie algebra elements from rotation matrices.
        
        :param a: Rotation matrices from which logarithm is to be computed
        :param b: Target rotation matrices
        :return: Skew-symmetric matrices (Lie algebra elements)
        """

        batch, f, n, _, _ = a.shape

        val = b @ torch.transpose(a, 3,4)
        
        val = val.view(-1, 3, 3)

        out = transforms.so3_log_map(val)

        
        return out.view(batch, f, n, 3)
    
    def dist(self, a, b):
        """
        Compute the relative angle between two sets of rotation matrices using the SO(3) distance metric.
        
        :param a: First set of rotation matrices
        :param b: Second set of rotation matrices
        :return: Tensor of distances
        """
        batch, f, n, _, _ = a.shape
        a_new = a.view(-1, 3, 3)
        b_new = b.view(-1, 3, 3)

        out = transforms.so3_relative_angle(a_new, b_new)

        return out.view(batch, f, n)

    def tangent_rotation_to_tangent(self, x):
        """
        Extract the tangent vector from a tangent rotation matrix representation.
        First the matricies are terned into Skew-symmetric matrices then the vector is extracted
        
        :param x: Tensor of tangent rotation matrices
        :return: Tangent vectors corresponding to the input matrices
        """

        squeue_sym = 0.5 * (x - torch.transpose(x, 3,4))
        
        theta_x = squeue_sym[:, :, :, 2, 1].unsqueeze(-1)
        theta_y = squeue_sym[:, :, :, 0, 2].unsqueeze(-1)
        theta_z = squeue_sym[:, :, :, 1, 0].unsqueeze(-1)

        return torch.cat([theta_x, theta_y, theta_z], dim=3)

    def conditional_flow(self, x_0, x_1, t):
        """
        Compute the conditional flow using Riemanian Flow Matching from paper "Flow Matching on General Geometries"
        
        :param x_0: Initial state rotation matrices
        :param x_1: Final state rotation matrices
        :param t: Time parameter for flow computation
        :return: The conditional flow as rotation matrices
        """
        log_x0_x1 = self.log_map(x_0, x_1)
        out = self.exp_map(x_0, t * log_x0_x1)
  
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
        d_0_1 = self.dist(x_0, x_1)
        d_t_1 = self.dist(x_t, x_1)

        grad_d_t_1 = torch.autograd.grad(
            inputs=x_t,
            outputs=d_t_1,
            grad_outputs=torch.ones_like(d_0_1),
            create_graph=True,
            retain_graph=True)[0]

        out = d_0_1[:, :, :, None,None] * grad_d_t_1 / ((torch.linalg.norm(grad_d_t_1, dim=(3,4))+epsilon)[:, :, :,None,None])
        
        if (out.isnan().any()):
            out = torch.nan_to_num(out, nan=0.0)
        
        out = self.tangent_rotation_to_tangent(out)

        return out

    def apply_nn(self, x, t):
        """
        Apply the neural network model to rotations to predict the conditional vector field
        Using 6D representation for neumerical stability
        
        :param x: Input rotation matrices
        :param t: Time parameter
        :return: Predicted changes in the state as rotation matrices
        """
        x_6d = transforms.matrix_to_rotation_6d(x)
        shape = x_6d.shape
        x_6d = x_6d.view(x.shape[0], x.shape[1], -1)

        v_t = self.model(x_6d, t)
        
        v_t = v_t.view(shape)
        v_t = transforms.rotation_6d_to_matrix(v_t)

        v_t = self.tangent_rotation_to_tangent(v_t)

        return v_t

    def train_step(self, x_1):
        """
        Perform a training step using input data x_1 to optimize the predictive model.
        
        :param x_1: Input data used for training
        :return: The computed loss as a result of training
        """
        x_1 = transforms.euler_angles_to_matrix(x_1, "XYZ")

        x_0 = self.gen_random_x(x_1)
        t = self.sample_timestep(x_0.shape[0]).requires_grad_(True)

        psi_t = self.conditional_flow(x_0, x_1, t[:, None, None])

        v_t = self.apply_nn(psi_t, t)

        con_vec = self.conditional_vector_field(x_0, psi_t, x_1, t)

        # weighting the loss by the timestep, so losses closer to 1 are more important than those close to 0
        # loss = torch.mean( t.squeeze(-1) * torch.linalg.norm(v_t + con_vec, dim=(1,2,3)) )
        loss = F.mse_loss(v_t, -con_vec)

        return loss

    def sample(self, n, frames=5, timesteps=50, scale=1):
        """
        Generate a sample path using the model's dynamics over specified timesteps.
        
        :param n: Number of samples to generate
        :param timesteps: Number of timesteps to simulate
        :param scale: Scaling factor for timestep computations
        :return: Final state after simulation
        """

        self.model.eval()
        with torch.no_grad():
            z = self.gen_random_x(torch.zeros(n, frames, 21, 3,3)).to(self.device)
            steps = torch.linspace(0.0, 1.0, timesteps, device=self.device)
            for i in range(timesteps):
                t = torch.full((n,1), steps[i], device=self.device)
                v_t = self.apply_nn(z,t)
                # z = z + 1 / timesteps * v_t
                z = self.exp_map(z, -(1/timesteps) * scale * v_t)
                
        self.model.train()

        z = transforms.matrix_to_axis_angle(z)

        return z

    def sample_full(self, n, frames=5, timesteps=50, scale=1):
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
            z = self.gen_random_x(torch.zeros(n, frames, 21, 3,3)).to(self.device)
            steps = torch.linspace(0.0, 1.0, timesteps, device=self.device)
            for i in range(timesteps):
                zs.append(transforms.matrix_to_axis_angle(z))
                t = torch.full((n,1), steps[i], device=self.device)
                v_t = self.apply_nn(z,t)
                # z = z + 1 / timesteps * v_t
                z = self.exp_map(z, -(1/timesteps) * scale * v_t)
                
        self.model.train()
        zs.append(transforms.matrix_to_axis_angle(z))
        zs = torch.stack(zs)

        return transforms.matrix_to_axis_angle(z), zs

    def sample_partial(self, partial_x_1, mask, frames=5, timesteps=50, scale=1):
        """
        Generate all the missing bones in the partial_x_1 using the model's dynamics over specified timesteps.
        Specifically by using Motion editing by sampling trajectory rewriting from the paper "Motion Flow Matching for Human Motion Synthesis and Editing"
        
        :param partial_x_1: A partial set of rotation matrices with random values for the missing values
        :param m: mask containing all the bones that are not missing
        :param timesteps: Number of timesteps to simulate
        :param scale: Scaling factor for timestep computations
        :return: Final state after simulation
        """

        self.model.eval()
        with torch.no_grad():
            x_1 = transforms.euler_angles_to_matrix(x_1, "XYZ").to(self.device)
            n = partial_x_1.shape[0]

            z_0 = self.gen_random_x(torch.zeros(n, frames, 21, 3,3)).to(self.device)
            steps = torch.linspace(0.0, 1.0, timesteps, device=self.device)

            z = z_0
            for i in range(timesteps):
                t = torch.full((n,1, 1), steps[i], device=self.device)

                
                partial_x_t = self.conditional_flow(z_0, partial_x_1, t)
                z[:, mask, :] = partial_x_t

                v_t = self.apply_nn(z,t)
                # z = z + 1 / timesteps * v_t
                z = self.exp_map(z, -(1/timesteps) * scale * v_t)
                
        self.model.train()

        z = transforms.matrix_to_axis_angle(z)

        return z

    def denoise_pose(self, noisy_pose, initial_timestep=30, timesteps=50, scale=1):
        """
        Denoise a noisy pose using the model's dynamics over specified timesteps.
        
        :param noisy_pose: Noisy pose to denoise
        :param initial_timestep: the point at which the denoising starts
        :param timesteps: Number of timesteps to simulate
        :param scale: Scaling factor for timestep computations
        :return: Final state after simulation
        """


        self.model.eval()
        with torch.no_grad():
            z = transforms.euler_angles_to_matrix(noisy_pose, "XYZ").to(self.device)
            n = z.shape[0]
            
            steps = torch.linspace(0.0, 1.0, timesteps, device=self.device)
            for i in range(initial_timestep, timesteps):
                t = torch.full((n,1), steps[i], device=self.device)
                v_t = self.apply_nn(z,t)
                # z = z + 1 / timesteps * v_t
                z = self.exp_map(z, -(1/timesteps) * scale * v_t)
                
        self.model.train()

        z = transforms.matrix_to_axis_angle(z)

        return z