# Image to 3D Generation

In order to generate 3D poses from images smplify is used with a specific prior as stated in the paper.
This section will explain how we specifically generate the poses, look to the paper for how the mathematics works.

## DPoser

We use the code from [DPoser](https://github.com/moonbow721/DPoser) to generate the final poses.

## STEP 1: Install Dependencys

First download the DPose git repository using and move to the v2 branch
```
git clone https://github.com/moonbow721/DPoser.git
cd ./DPoser
git checkout v2
```

Then download ViTPose using
```
git clone https://github.com/ViTAE-Transformer/ViTPose.git
```

Create a new Python Environment and download the specific requirements following both of there instructions

(Personally I used torch 13.1 with CUDA 11.7 and that seemed to work).


## STEP 2: Generate 2D Poses

Please follow the "2D Human Pose Top-Down Video Demo" using mmdet demo in order to generate the specific 2D bounding boxes for the 3DPW dataset images.

File `/demo/top_down_img_demo_with_mmdet.py`

We modified the code a small amount and that will be in the modified Section (`/docs/Image_to_3d_code/ViTPose`)

Command: `python demo/top_down_img_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth --img-root /mnt/c/Users/mikew/GitHub/3DPW/imageFiles/test/ --output_folder "/mnt/c/Users/mikew/GitHub/3DPW/vit_poses/"`


## STEP 3: Modify DPose to use a new Prior

Several Files need to be modified in DPoser all of which are stored in (`/docs/Image_to_3d_code/DPoser`)

NOTE: if you also want to run any other priors please download the best models from the respective repositories into their location in DPoser `/other_priors/PRIOR`
- [VPoser](https://smpl-x.is.tue.mpg.de/index.html) (NEED TO MAKE AN ACCOUNT)
- [Pose-NDF](https://github.com/garvita-tiwari/PoseNDF/tree/version2)
- [NRDF](https://github.com/hynann/NRDF)

## STEP 4: Generate Poses



# Acknowlegements

#### DPose
```
@article{lu2023dposer,
  title={DPoser: Diffusion Model as Robust 3D Human Pose Prior},
  author={Lu, Junzhe and Lin, Jing and Dou, Hongkun and Zeng, Ailing and Deng, Yue and Zhang, Yulun and Wang, Haoqian},
  journal={arXiv preprint arXiv:2312.05541},
  year={2023}
}
```

#### ViTPose

```
@inproceedings{
  xu2022vitpose,
  title={Vi{TP}ose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}
```

####
```
@inproceedings{Bogo:ECCV:2016,
    title = {Keep it {SMPL}: Automatic Estimation of {3D} Human Pose and Shape
    from a Single Image},
    author = {Bogo, Federica and Kanazawa, Angjoo and Lassner, Christoph and
    Gehler, Peter and Romero, Javier and Black, Michael J.},
    booktitle = {Computer Vision -- ECCV 2016},
    series = {Lecture Notes in Computer Science},
    publisher = {Springer International Publishing},
    month = oct,
    year = {2016}
}
```