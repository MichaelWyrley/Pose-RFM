# This is modified code from https://github.com/ViTAE-Transformer/ViTPose/blob/main/demo/top_down_img_demo_with_mmdet.py
# Please replace the file in that directory if you need to run inference

# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import glob
import numpy as np

def inference():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    '''
    python demo/top_down_img_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth --img-root /mnt/c/Users/mikew/GitHub/3DPW/imageFiles/test/ --output_folder "/mnt/c/Users/mikew/GitHub/3DPW/vit_poses/"
    '''

    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection', default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection', default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
    parser.add_argument('pose_config', help='Config file for pose', default='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose', default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--output_folder', type=str, default='', help='The folder to save the output')
    
    # parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # assert args.show or (args.out_img_root != '')
    # assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # image_name = os.path.join(args.img_root, args.img)

    directories = os.listdir(args.img_root)
    for d in directories:
        files = sorted(glob.glob(args.img_root + d + "/*.jpg"))

        print("calculating for: " + d)
        pose_body = {
            'bbox': [],
            'keypoints': [],

            # 'pose_results': [],
            # # 'returned_outputs': []
        }

        output_dir = args.output_folder + d
        for (i,image_name) in enumerate(files):
            print("calculating frame: ", i)

            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, image_name)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            # test a single image, with a list of bboxes.

            # optional
            return_heatmap = False

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None

            '''
            tuple:
                - pose_results (list[dict]): The bbox & pose info. \
                    Each item in the list is a dictionary, \
                    containing the bbox: (left, top, right, bottom, [score]) \
                    and the pose (ndarray[Kx3]): x, y, score.
                - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
                    torch.Tensor[N, K, H, W]]]): \
                    Output feature maps from layers specified in `outputs`. \
                    Includes 'heatmap' if `return_heatmap` is True.
            '''
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                image_name,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)


            pose_body['bbox'].append(pose_results[0]['bbox'])
            pose_body['keypoints'].append(pose_results[0]['keypoints'])
            # pose_body['returned_outputs'].append(returned_outputs)

        np.savez(output_dir, **pose_body)


if __name__ == '__main__':
    inference()
