echo "starting python"

cd ./transfer_model

mkdir ../transfer_data/meshes/smplx/downtown_stairs_00
mkdir ../transfer_data/meshes/smplx/downtown_walkUphill_00
mkdir ../transfer_data/meshes/smplx/flat_guitar_01
mkdir ../transfer_data/meshes/smplx/flat_packBags_00
mkdir ../transfer_data/meshes/smplx/outdoors_fencing_01

python write_obj.py --model-folder <Path to smpl and smplx model files>/human_model_files/ --motion-file <Path to Pose-RFM>/Pose-RFM/dataset/3DPW/smplx_poses/ --output-folder ../transfer_data/meshes/smplx/ --model-type smplx

cd ../

python -m transfer_model --exp-cfg config_files/files/downtown_stairs_00.yaml
mkdir ./transfer_data/meshes/smpl/downtown_stairs_00
mv ./output/* ./transfer_data/meshes/smpl/downtown_stairs_00/

python -m transfer_model --exp-cfg config_files/files/downtown_walkUphill_00.yaml
mkdir ./transfer_data/meshes/smpl/downtown_walkUphill_00
mv ./output/* ./transfer_data/meshes/smpl/downtown_walkUphill_00/

python -m transfer_model --exp-cfg config_files/files/flat_guitar_01.yaml
mkdir ./transfer_data/meshes/smpl/flat_guitar_01
mv ./output/* ./transfer_data/meshes/smpl/flat_guitar_01/

python -m transfer_model --exp-cfg config_files/files/flat_packBags_00.yaml
mkdir ./transfer_data/meshes/smpl/flat_packBags_00
mv ./output/* ./transfer_data/meshes/smpl/flat_packBags_00/

python -m transfer_model --exp-cfg config_files/files/outdoors_fencing_01.yaml
mkdir ./transfer_data/meshes/smpl/outdoors_fencing_01
mv ./output/* ./transfer_data/meshes/smpl/outdoors_fencing_01/

python ./transfer_model/merge_output.py --output_dir <Path to Pose-RFM>/dataset/3DPW/npz_poses/generated_smpl --input_dir transfer_data/meshes/smpl --gender neutral
