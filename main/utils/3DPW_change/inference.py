import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
import cv2
from tqdm import tqdm
import json
from typing import Literal, Union
from mmdet.apis import init_detector, inference_detector
from utils.inference_utils import process_mmdet_results, non_max_suppression



import glob



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--pretrained_model', type=str, default='smpler_x_s32')
    parser.add_argument('--testset', type=str, default='<PATH TO Pose-RFM root>/dataset/3DPW/imageFiles/test/')
    parser.add_argument('--agora_benchmark', type=str, default='na')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--output_folder', type=str, default='<PATH TO Pose-RFM root>/dataset/3DPW/smplx_poses/')
    parser.add_argument('--demo_dataset', type=str, default='na')
    parser.add_argument('--demo_scene', type=str, default='all')
    parser.add_argument('--show_verts', action="store_true")
    parser.add_argument('--show_bbox', action="store_true")
    parser.add_argument('--save_mesh', action="store_true")
    parser.add_argument('--multi_person', action="store_true")
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--bbox_thr', type=int, default=50)
    args = parser.parse_args()
    return args

# /vol/bitbucket/mew23/individual-project/dataset/3DPW/imageFiles/test/


def gen_data():

    args = parse_args()
    config_path = osp.join('./config', f'config_{args.pretrained_model}.py')
    ckpt_path = osp.join('../pretrained_models', f'{args.pretrained_model}.pth.tar')

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(args.testset, args.agora_benchmark, shapy_eval_split=None, 
                            pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True

    # load model
    from base import Demoer
    from utils.preprocessing import load_img, process_bbox, generate_patch_image
    from utils.vis import render_mesh, save_obj
    from utils.human_models import smpl
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()

    ### mmdet init
    checkpoint_file = '../pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    config_file= '../pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'


    directories = os.listdir(args.testset)
    for d in directories:
        files = glob.glob(args.testset + d + "/*.jpg")

        print("calculating for: " + d)
        pose_body = {
            'global_orient': [],
            'body_pose': [],
            'left_hand_pose': [],
            'right_hand_pose': [],
            'jaw_pose': [],
            'leye_pose': [],
            'reye_pose': [],
            'betas': [],
            'expression': [],
            'transl': [],
        }

        output_dir = args.output_folder + d

        for frame in files:

            # prepare input image
            transform = transforms.ToTensor()
            original_img = load_img(frame)
            vis_img = original_img.copy()
            original_img_height, original_img_width = original_img.shape[:2]
            # os.makedirs(output_folder, exist_ok=True)

            ## mmdet inference
            mmdet_results = inference_detector(model, frame)
            mmdet_box = process_mmdet_results(mmdet_results, cat_id=0, multi_person=True)
            
            ## mmdet inference
            print(len(mmdet_box))
            # mmdet_box = non_max_suppression(mmdet_box[0], args.iou_thr)
            # num_bbox = len(mmdet_box)
            num_bbox = 1
            mmdet_box = mmdet_box[0]

            if(len(mmdet_box) == 0):
                print(args.output_folder + d)
            
            ## loop all detected bboxes
            for bbox_id in range(num_bbox):
                mmdet_box_xywh = np.zeros((4))
                mmdet_box_xywh[0] = mmdet_box[bbox_id][0]
                mmdet_box_xywh[1] = mmdet_box[bbox_id][1]
                mmdet_box_xywh[2] =  abs(mmdet_box[bbox_id][2]-mmdet_box[bbox_id][0])
                mmdet_box_xywh[3] =  abs(mmdet_box[bbox_id][3]-mmdet_box[bbox_id][1]) 

                # skip small bboxes by bbox_thr in pixel
                if mmdet_box_xywh[2] < args.bbox_thr or mmdet_box_xywh[3] < args.bbox_thr * 3:
                    continue

                bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height)
                img, _, _ = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
                img = transform(img.astype(np.float32))/255
                img = img.cuda()[None,:,:,:]
                inputs = {'img': img}
                targets = {}
                meta_info = {}

                # mesh recovery
                with torch.no_grad():
                    out = demoer.model(inputs, targets, meta_info, 'test')

                
                pose_body['global_orient'].append(out['smplx_root_pose'].reshape(-1,3).cpu().numpy())
                pose_body['body_pose'].append(out['smplx_body_pose'].reshape(-1,3).cpu().numpy())
                pose_body['left_hand_pose'].append(out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy())
                pose_body['right_hand_pose'].append(out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy())
                pose_body['jaw_pose'].append(out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy())
                pose_body['leye_pose'].append(np.zeros((1, 3)))
                pose_body['reye_pose'].append(np.zeros((1, 3)))
                pose_body['betas'].append(out['smplx_shape'].reshape(-1,10).cpu().numpy())
                pose_body['expression'].append(out['smplx_expr'].reshape(-1,10).cpu().numpy())
                pose_body['transl'].append( out['cam_trans'].reshape(-1,3).cpu().numpy())
        
        np.savez(output_dir, **pose_body)

if __name__ == "__main__":
    print("Starting inference")
    gen_data()