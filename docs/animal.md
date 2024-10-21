# Training and Generating Animal Poses

## Training Animal Model

Training the animal model is very similar to training the Human Pose Model, except there are categories that can be used to condition it.
Inorder to train the model `main/animal/train_model.py` can be used in the same fation as `main/train.py` is used


### Visualising Animal Data

In order to visualise the data `main/animal/vis_animal.py` can be used.
The specific labels used is stored in the `categories` parameter in the .npz file.
These will then be converted into actual animal models via the `label_to_betas` parameter, where each label (super category) will be converted into a specific animal (category) from the model data.

### Generating and Visualising the Data

NOTE The current best animal uses FlowMatchingMatrix_OLD which had a bug resulting in alright performance but not the best
If you want to use your own pre-trained model please change gen_poses.py to use the newer one!!!

In order to generate and visualise the data `main/animal/gen_poses.py` can be used.
The specific labels used to generate the data can be specified by changeing the `label` variable on line 82.

Change line 84 to sample the full tragetory (sample_grid_video) or sample the final poses (sample_grid)


