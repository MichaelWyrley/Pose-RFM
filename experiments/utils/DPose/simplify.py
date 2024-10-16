# Modified from https://github.com/moonbow721/DPoser/blob/v2/run/tester/body/smplify.py
import sys
sys.path.append('')

import torch
from torch import nn

from experiments.utils.DPose.constants import JOINT_IDS
from experiments.utils.DPose.fitting_losses import camera_fitting_loss, body_fitting_loss, perspective_projection

from experiments.utils.DPose.transforms import flip_orientations
from human_body_prior.body_model.body_model import BodyModel

from experiments.utils.DPose.smpl import SMPL
from experiments.utils.DPose.constants import BEND_POSE_PATH
import numpy as np



import cv2
import glob


class SMPLify:
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 cam_intrinsics,
                 refine_model=None,
                 step_size=1e-2,
                 batch_size=32,
                 num_iters_cam=100,
                 num_iters_pose=100,
                 side_view_thsh=25.0,
                 device='cuda',
                 args=None):
        # Store options
        self.device = device
        self.cam_intrinsics = cam_intrinsics
        self.side_view_thsh = side_view_thsh
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip']
        self.ign_joints = [JOINT_IDS[i] for i in ign_joints]
        self.num_iters_cam = num_iters_cam
        self.num_iters_pose = num_iters_pose
        self.args = args
        # self.prior_name = args.prior

        if args['pose_prior'] == 'Pose-RFM':
            from experiments.utils.DPose.prior import Pose_RFM
            self.pose_prior = Pose_RFM(batch_size, args.config_path, args)

        # if args['pose_prior'] == 'DPoser':
        #     self.pose_prior = DPoser(batch_size, args.config_path, args)
        #     self.time_strategy = args.time_strategy
        #     self.t_max = 0.12
        #     self.t_min = 0.08
        #     self.fixed_t = 0.10
        # elif args['pose_prior'] == 'GMM':
        #     from lib.body_model.prior import MaxMixturePrior
        #     self.pose_prior = MaxMixturePrior(prior_folder=constants.GMM_WEIGHTS_DIR,
        #                                       num_gaussians=8,
        #                                       dtype=torch.float32).to(self.device)
        # elif args['pose_prior'] == 'VPoser':
        #     from lib.body_model.prior import VPoser, VPoser_new
        #     support_dir = '/data3/ljz24/projects/3d/human_body_prior/support_data/dowloads'
        #     self.pose_prior = VPoser(support_dir).to(self.device)
        #     # config_path = 'subprior.configs.body.optim.set1.get_config'
        #     # self.pose_prior = VPoser_new(config_path).to(self.device)
        # elif args['pose_prior'] == 'Posendf':
        #     from lib.body_model.prior import Posendf
        #     config = '/data3/ljz24/projects/3d/PoseNDF/checkpoints/config.yaml'
        #     ckpt = '/data3/ljz24/projects/3d/PoseNDF/checkpoints/checkpoint_v2.tar'
        #     self.pose_prior = Posendf(config, ckpt).to(self.device)
        else:
            self.pose_prior = None


        self.time_strategy = args['time_strategies']
        self.t_max = 0.12
        self.t_min = 0.08
        self.fixed_t = 0.10

        # self.loss_weights = {'pose_prior_weight': [50, 20, 10, 5, 2],
        #                      'shape_prior_weight': [50, 20, 10, 5, 2],
        #                      'angle_prior_weight': [150, 50, 30, 15, 5],
        #                      'coll_loss_weight': [0, 0, 0, 0.01, 1.0],
        #                      }
        self.loss_weights = {'pose_prior_weight': [20, 10, 5, 2],
                             'shape_prior_weight': [20, 10, 5, 2],
                             'angle_prior_weight': [50, 30, 15, 5],
                             'coll_loss_weight': [0, 0, 0.01, 1.0],
                             }
        self.stages = len(self.loss_weights['pose_prior_weight'])

        # self.body_model = BodyModel(args['model'], num_betas=10, model_type='smplh').to(device)
        self.body_model = SMPL(model_path=args['model'], batch_size=batch_size).to(device)

    def pose_to_vert(self, pose_body, betas, global_orient, trans, args):
        time_length = len(pose_body)

        pose_body = pose_body.reshape(time_length, -1)
        
        # body_pose_beta = body_model(pose_body=pose_body, betas=betas, root_orient = global_orient, trans=trans)
        body_pose_beta = self.body_model(betas=betas,
                                        body_pose=pose_body,
                                        global_orient=global_orient,
                                        pose2rot=True,
                                        transl=trans)

        return body_pose_beta

    def sample_continuous_time(self, iteration):
        total_steps = self.stages * self.num_iters_pose

        if self.time_strategy == '1':
            t = self.pose_prior.eps + torch.rand(1, device=self.device) * (self.pose_prior.sde.T - self.pose_prior.eps)
        elif self.time_strategy == '2':
            t = torch.tensor(self.fixed_t)
        elif self.time_strategy == '3':
            t = self.t_min + torch.tensor(total_steps - iteration - 1) / total_steps * (self.t_max - self.t_min)
        else:
            t = 0

        return t

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # smpl_poses = self.body_model.mean_poses[:24 * 3].unsqueeze(0).repeat(init_pose.shape[0], 1).to(self.device)  # N*66
        # global_orient = smpl_poses[:, :3].clone()
        # # bend_pose = torch.from_numpy(np.load(BEND_POSE_PATH)['pose'][:, :24 * 3]).to(self.device)
        # # smpl_poses[bend_init, 3:] = bend_pose[:, 3:]


        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2].clone()
        joints_conf = keypoints_2d[:, :, -1].clone()

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        # global_orient = torch.zeros(init_pose.shape[0], 3, device=init_pose.device)

        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        # camera_opt_params = [global_orient, camera_translation]
        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters_cam):
            smpl_output = self.pose_to_vert(betas=betas,
                                        pose_body=body_pose,
                                        global_orient=global_orient,
                                        trans=camera_translation,
                                        args=self.args)

            model_joints = smpl_output.joints
            loss = camera_fitting_loss(model_joints, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf, cam_intrinsics=self.cam_intrinsics, part='body')
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()



        left_shoulder_idx, right_shoulder_idx = 2, 5
        shoulder_dist = torch.dist(joints_2d[:, left_shoulder_idx],
                                   joints_2d[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < self.side_view_thsh

        # Step 2: Optimize body joints
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.
        inputs = (global_orient, body_pose, betas, camera_translation, camera_center, joints_2d, joints_conf)

        if try_both_orient:
            pose, betas, camera_translation, reprojection_loss = self.optimize_and_compare(*inputs)
        else:
            reprojection_loss, (pose, betas, camera_translation, _) = self.optimize_body(*inputs)

        return pose, betas, camera_translation, reprojection_loss

    def optimize_and_compare(self, global_orient, body_pose, betas, camera_translation, camera_center, joints_2d,
                             joints_conf):
        original_loss, original_results = self.optimize_body(global_orient.detach(), body_pose, betas, camera_translation,
                                                             camera_center, joints_2d, joints_conf)
        flipped_loss, flipped_results = self.optimize_body(flip_orientations(global_orient).detach(), body_pose, betas,
                                                           camera_translation, camera_center, joints_2d, joints_conf)

        min_loss_indices = original_loss < flipped_loss  # [N,]

        pose = torch.where(min_loss_indices.unsqueeze(-1), original_results[0], flipped_results[0])
        betas = torch.where(min_loss_indices.unsqueeze(-1), original_results[1], flipped_results[1])
        camera_translation = torch.where(min_loss_indices.unsqueeze(-1), original_results[2], flipped_results[2])
        reprojection_loss = torch.where(min_loss_indices, original_loss, flipped_loss)

        return pose, betas, camera_translation, reprojection_loss

    def optimize_body(self, global_orient, body_pose, betas, camera_translation, camera_center, joints_2d, joints_conf):
        """
        Optimize only the body pose and global orientation of the body
        """
        batch_size = global_orient.shape[0]

        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        stage_weights = [dict(zip(self.loss_weights.keys(), vals)) for vals in zip(*self.loss_weights.values())]

        # for stage, current_weights in enumerate(tqdm(stage_weights, desc='Stage')):
        for stage, current_weights in enumerate(stage_weights):
            for i in range(self.num_iters_pose):
                body_optimizer.zero_grad()

                smpl_output = self.pose_to_vert(betas=betas,
                                        pose_body=body_pose,
                                        global_orient=global_orient,
                                        trans=camera_translation,
                                        args=self.args)

                model_joints = smpl_output.joints

                t = self.sample_continuous_time(iteration=stage * self.num_iters_pose + i)

                loss = body_fitting_loss(body_pose, betas, model_joints,
                                         joints_2d, joints_conf, self.pose_prior, t=t,
                                         cam_intrinsics=self.cam_intrinsics,
                                         **current_weights)


                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.pose_to_vert(betas=betas,
                                        pose_body=body_pose,
                                        global_orient=global_orient,
                                        trans=camera_translation,
                                        args=self.args)

            model_joints = smpl_output.joints
            t = self.sample_continuous_time(iteration=stage * self.num_iters_pose + i)
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints,
                                                  joints_2d, joints_conf, self.pose_prior, t=t,
                                                  cam_intrinsics=self.cam_intrinsics,
                                                  output='reprojection')

        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()
        return reprojection_loss, (pose, betas, camera_translation, reprojection_loss)

    # If the images are needed to be output
    def output_image(self, body_pose, betas, camera_translation, joints_2d, diff, img_loc, start_frame, save_loc):
        smpl_output = self.pose_to_vert(betas=betas,
                                        pose_body=body_pose[:, 3:],
                                        global_orient=body_pose[:, :3],
                                        trans=camera_translation,
                                        args=self.args)

        model_joints = smpl_output.joints
        rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(model_joints.shape[0], -1, -1)
        projected_joints = perspective_projection(model_joints, rotation,
                                              self.cam_intrinsics)[:, :25]
        
        image_seq = sorted(glob.glob(img_loc + "/*.jpg"))[start_frame:start_frame+betas.shape[0]]
        colour = np.random.randint(255, size=(25,3), dtype=int)
        colour = colour.tolist()
        print("Saving images to:", img_loc)
        for k, img_loc in enumerate(image_seq):
            image = cv2.imread(img_loc)

            
            for i in range(25):
                image = cv2.circle(image, (joints_2d[k,i,:2] + diff[k]).cpu().detach().numpy().astype(int), 5, colour[i], 2)
                image = cv2.circle(image, (projected_joints[k,i] + diff[k]).cpu().detach().numpy().astype(int), 10, colour[i], 2)
                # image = cv2.circle(image, joints_2d.cpu().detach().numpy().astype(int)[0][i], 10, (0,255,0), 2)
                # image = cv2.circle(image, projected_joints.cpu().detach().numpy().astype(int)[0][i], 10, (0,0,255), 2)

            save_loc_name = img_loc.split('/')[-1]
            print("saving image:", save_loc_name)
            cv2.imwrite(save_loc + '/' + save_loc_name, image)