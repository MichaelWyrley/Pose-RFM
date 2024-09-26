# Pose Generation


## Generating Poses

Generating samples is done using the `main/utils/sampling/sample.py` file
You can sample a the final generated poses using `sample_single`, the full list of poses as they are denoised where each output file contains only that poses information using `sample_full` or the full list of poses as they are denoised where each file contains a single denoising timestep using `sample_full_for_video`

They all take the flow matching model instance (which should contain the Vector Fields Neural Network) and different arguments:
- 'samples': The number of samples wanting to be generated
- 'sample_timestep': The amount of timesteps to denoise the model
- 'print': Output information about the generation process
- 'load_model': The directory of the model (relative to the 'directory' value)
- 'scale': The scale used to denoise the value
- 'sample_dir': The place to save the samples (relative to the 'directory' value)

## Generating Images From Poses

You can visualise your generated poses using either `main/utils/image/visualise.py` or `main/utils/image/visualise_torch3d.py` file.
`main/utils/image/visualise.py` produces results slightly quicker from testing wheres `main/utils/image/visualise_torch3d.py` allows for the output of obj files
It has a function called `visualise` which will read in the poses and ouput .png files of them, it takes arguments:

- 'frame': The pose file wanting to be visualised (relative to the 'directory' value)
- 'model': The location of the SMPL model wanting to be used (All images in project generated using nutral model)
- 'image_loc': The place to save the images (relative to the 'directory' value)
- 'name': An extra naming parameter for grids if needed (default should be '')
- 'print': Output information about the generation process
- 'time_length': The amount of poses that need to be generated (if you want to generate all poses in a file set to a value greater than the number of poses in the file)
- 'output_obj': Boolean value to allow the ouput of obj files if using visualise_torch3d.py,
- 'save_grid': Boolean value to say if you should save the images as single images or as a grid of images

## Generating Both at the same Time

If you want to generate the samples and images at the same time then use `gen_images.py`.

It has functions to generate a single grid or image of the output (sample_grid), generate a the full denoising values as grids or images (sample_video), or sample the outputs in a manour allowing for a video of denoising to be generated using (sample_grid_video)

They all take the flow matching model instance (which should contain the Vector Fields Neural Network) and different arguments:
- 'samples': The number of samples wanting to be generated
- 'sample_timestep': The amount of timesteps to denoise the model
- 'load_model': The directory of the model (relative to the 'directory' value)
- 'scale': The scale used to denoise the value
- 'sample_dir': The place to save the samples (relative to the 'directory' value)
- 'name': An extra naming parameter for grids if needed (default should be '')
- 'frame': The pose file wanting to be visualised (relative to the 'directory' value)
- 'model': The location of the SMPL model wanting to be used (All images in project generated using nutral model)
- 'image_loc': The place to save the images (relative to the 'directory' value)
- 'print': Output information about the generation process
- 'time_length': The amount of poses that need to be generated (if you want to generate all poses in a file set to a value greater than the number of poses in the file)
- 'output_obj': Boolean value to allow the ouput of obj files if using visualise_tor
- 'save_grid': Boolean value to say if you should save the images as single images or as a grid of images
- 'sample_single': Sample the final timestep

## Generating a video from the files

The sequence of images generated can be turned into a video using ffmpeg
`ffmpeg -framerate 18 -pattern_type glob -i 'IMGs_LOCATION/*.png'   -c:v libx264 -pix_fmt yuv420p out.mp4`

# Pose Denoising

Pose denoising can be done using the `denoise_pose` function in the flow matching class it takes in the noisy pose, the intitial timestep to start denoising at, the final timestep to end denoising and the scale to take each denoising step.

The file `utils/sampling/pose_denoising.py` outputs denoised poses based on the poses in the training dataset. It uses the flow matching object to noise the initial clean pose some amount then denoises it.
It takes the flow matching object and arguments:
- 'clean': Location of the clean data (relative to the 'directory')
- 'batch_size': Amount of data wanting to be processed (set to 1 usually)
- 'no_samples': The amount of samples to generate
- 'scale': The scale of denoising
- 'initial_timestep': The initial timestep wanting to start the denoising at
- 'timesteps': The final timestep to end denoising
- 'load_model': The directory of the model (relative to the 'directory' value)
- 'sample_dir': The place to save the samples (relative to the 'directory' value)
- 'frame': The pose file wanting to be visualised (relative to the 'directory' value)
- 'image_loc': The place to save the images (relative to the 'directory' value)
- 'model': The location of the SMPL model wanting to be used (All images in project generated using nutral model)
- 'name': An extra naming parameter for grids if needed (default should be '')
- 'print': Output information about the generation process
- 'time_length': The amount of poses that need to be generated (if you want to generate all poses in a file set to a value greater than the number of poses in the file)
- 'output_obj': Boolean value to allow the ouput of obj files if using visualise_tor
- 'save_grid': Boolean value to say if you should save the images as single images or as a grid of images

# Masked Joint Generation

Masked Generation can be done using the `partial_corruption` function in the flow matching class it takes in the partial pose which has all joints available (add noise to missing joints if needed), the masked joints where 1 represents available and 0 represents not, the amount of denoising steps, the scaling for denoising

The file `utils/sampling/partial_generation.py` outputs Masked Generation poses based on the poses in the training dataset. It generates a random mask based on some masking threshold, then removes the joints that would be missing be replacing them with noise.
It takes the flow matching object and arguments:
- 'clean': Location of the clean data (relative to the 'directory')
- 'batch_size': Amount of data wanting to be processed (set to 1 usually)
- 'no_samples': The amount of samples to generate
- 'scale': The scale of denoising
- 'removal_level': The percentage of the body that should be masked
- 'timesteps': The final timestep to end denoising
- 'load_model': The directory of the model (relative to the 'directory' value)
- 'sample_dir': The place to save the samples (relative to the 'directory' value)
- 'frame': The pose file wanting to be visualised (relative to the 'directory' value)
- 'image_loc': The place to save the images (relative to the 'directory' value)
- 'model': The location of the SMPL model wanting to be used (All images in project generated using nutral model)
- 'name': An extra naming parameter for grids if needed (default should be '')
- 'print': Output information about the generation process
- 'time_length': The amount of poses that need to be generated (if you want to generate all poses in a file set to a value greater than the number of poses in the file)
- 'output_obj': Boolean value to allow the ouput of obj files if using visualise_tor
- 'save_grid': Boolean value to say if you should save the images as single images or as a grid of images
