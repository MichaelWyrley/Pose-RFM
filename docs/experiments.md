There are several tests that were run in the Pose-RFM paper including:
-   General Generation Performance
-   Generating Missing Joints Performance
-   Denoising Poses Performance
-   Image to 3D generation
-   Different Ablation Study Results

All experiments can be found in the `experiments` folder.
With the results from the paper in `current_tests` and graphs in `graphs`


`experiments\tests\dataset_mean_cov.npz` was taken from [NRDF](https://virtualhumans.mpi-inf.mpg.de/nrdf/)

Specific tests can be run under the `test` folder with graphs generated from `utils/gen_graphs.py`.

The General Generation Performance, Generating Missing Joints Performance, Denoising Poses Performance can be run from a single file using `experiments/test_suite.py`

The ablation study results can be generated from `experiments/test_suite.py` with the `load_model` argument needing to be changed in order to run different trained models.

Generating 3D poses from images can be run using `tests/test_image_to_3d.py` however the data needed to test this needs to be collected.
An explenation of this can be found [here](/docs/datasets.md)


