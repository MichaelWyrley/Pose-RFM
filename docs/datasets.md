This project uses three datasets and two different body models

# Datasets

## AMASS

If you want to train your own model on the dataset used in the paper then it can be accessed from the AMASS [website](https://amass.is.tue.mpg.de/)

The file structure and preprocessing steps used are the same as in NRDF so follow the steps specified there [website](https://github.com/hynann/NRDF/blob/master/docs/data.md)

The `<DATA_ROOT>` directory is `dataset/amass`

This project uses the [body_visualizer](https://github.com/nghorbani/body_visualizer) and [human_body_prior](https://github.com/nghorbani/human_body_prior) packages to render and generate the poses, however the [smplx](https://github.com/vchoutas/smplx) package can also be used

The body models should be stored under `dataset/models/SMPL`

## Animal3d

The SMAL models can be downloaded from [here](https://smal.is.tue.mpg.de/index.html) and should be placed in the `dataset/models/SMAL` directory

The animal3d dataset can be downloaded from [here](https://xujiacong.github.io/Animal3D/).
Please then run `main/animal/sample_dataset_poses.py` with arguments:
```
args = {
    'save': 'dataset/animal3d/SAMPLED_POSES/test',
    'data': 'dataset/animal3d/test.json'
}
```
and  
```   
args = {
        'save': 'dataset/animal3d/SAMPLED_POSES/train',
        'data': 'dataset/animal3d/train.json'
    }
```
To generate the npz files of the dataset.

## 3DPW

The dataset can be downloaded from [here](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

Please extract the imageFiles.zip and sequenceFiles.zip into the `datasets/3DPW` directory

Please make the directory tree look like
```
3DPW
└───imageFiles
│   └───ignore
│       └───all other files
│   └───test
│        └───downtown_stairs_00
│        └───downtown_walkUphill_00
│        └───flat_guitar_01
│        └───flat_packBags_00
│        └───outdoors_fencing_01
|
└───sequenceFiles
│   └───ignore
│       └───all other files
│   └───test
│        └───downtown_stairs_00.pkl
│        └───downtown_walkUphill_00.pkl
│        └───flat_guitar_01.pkl
│        └───flat_packBags_00.pkl
│        └───outdoors_fencing_01.pkl
│   
└───npz_poses
    └───ground_truth
         └───downtown_stairs_00.npz
         └───downtown_walkUphill_00.npz
         └───flat_guitar_01.npz
         └───flat_packBags_00.npz
         └───outdoors_fencing_01.npz
    └───generated_smpl
    └───denoised_smpl
|
└───smplx_poses
```

Next generate all the SMPLer-x predictions, download SMPLer-X from [here](https://github.com/caizhongang/SMPLer-X) and create a new virtual environment for it.
Install all the required packages.
Replace `main/inference.py` in SMPLer-x with `main/utils/3DPW_change/inference.py` from Pose-RFM and change lines 31 and 36 to what is needed then run it.
This will generate all smplx predictions for the images.

To convert the smplx predictions into smpl ones download SMPLx from [here](https://github.com/vchoutas/smplx) and create a new virtual environment for it.
Install all the required packages and copy the required files
Then follow the steps in [here](https://github.com/vchoutas/smplx/tree/main/transfer_model) to generate the .obj files.
Specifically running the `transfer_model/write_obj.py` file for all the required smplx outputs

After the obj s are generated they can be transfered into smpl models by doing:

create config files for all test cases by changing the following parameters
```
dataset: mesh_folder: data_folder: 'smplx/transfer_data/meshes/smplx/downtown_stairs_00'
```

Then copy `main/utils/3DPW/run_smplx.sh` from Pose-RFM to the root directory of the smplx folder change line 23 to what is needed and run it.



# bibtexts for all datasets

#### AMASS
```
@conference{AMASS:ICCV:2019,
  title = {{AMASS}: Archive of Motion Capture as Surface Shapes},
  author = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {International Conference on Computer Vision},
  pages = {5442--5451},
  month = oct,
  year = {2019},
  month_numeric = {10}
}
```


#### Animal3d
```
@article{xu2023animal3d,
  title={Animal3D: A Comprehensive Dataset of 3D Animal Pose and Shape},
  author={Xu, Jiacong and Zhang, Yi and Peng, Jiawei and Ma, Wufei and Jesslen, Artur and Ji, Pengliang and Hu, Qixin and Zhang, Jiehua and Liu, Qihao and Wang, Jiahao and others},
  journal={arXiv preprint arXiv:2308.11737},
  year={2023}
}
```

#### 3DPW

```
@inproceedings{vonMarcard2018,
    title = {Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera},
    author = {von Marcard, Timo and Henschel, Roberto and Black, Michael and Rosenhahn, Bodo and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018},
    month = {sep}
    }
```

#### SMPL
```
@article{SMPL:2015,
      author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
      title = {{SMPL}: A Skinned Multi-Person Linear Model},
      journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
      month = oct,
      number = {6},
      pages = {248:1--248:16},
      publisher = {ACM},
      volume = {34},
      year = {2015}
    }
```

#### SMAL

```
@inproceedings{Zuffi:CVPR:2017,
        title = {{3D} Menagerie: Modeling the {3D} Shape and Pose of Animals},
        author = {Zuffi, Silvia and Kanazawa, Angjoo and Jacobs, David and Black, Michael J.},
        booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        month = jul,
        year = {2017},
        month_numeric = {7}
      }
```