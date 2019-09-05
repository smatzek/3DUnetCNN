# 3D U-Net Convolution Neural Network with Keras

## Overview of tflmsv2 branch
The changes in the tflmsv2 branch from the original ellisdg repository generally consist of these items:
1. Changes to this README to expand on setup and usage sections.
2. Changes to the model to use tf.keras from TensorFlow.
3. Changes to use optionally use TensorFlow Large Model Support in IBM PowerAI,
 [IBM PowerAI documentation](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/welcome/welcome.html).
4. Changes to allow command line specification of TensorFlow Large Model
Support tuning parameters to train_isensee2017.py.
5. Changes to optionally allow training the model in a multi-GPU distributed
fashion using IBM Distributed Deep Learning.
6. Changes to optionally allow training the model in a multi-GPU distributed
fashion using [Horovod](https://github.com/uber/horovod)
7. Changes to enable CUDA profiling.

![Tumor Segmentation Example](doc/tumor_segmentation_illusatration.gif)
## Background
Originally designed after [this paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) on
volumetric segmentation with a 3D U-Net.
The code was written to be trained using the
[BRATS](http://www.med.upenn.edu/sbia/brats2017.html) data set for brain tumors, but it can
be easily modified to be used in other 3D applications.

## Tutorial using BRATS Data
### Training
1. Download the BRATS 2017 [GBM](https://app.box.com/shared/static/l5zoa0bjp1pigpgcgakup83pzadm6wxs.zip) and
[LGG](https://app.box.com/shared/static/x75fzof83mmomea2yy9kshzj3tr9zni3.zip) data. Place the unzipped folders in the
```brats/data/original``` folder.
2. Build dependencies

The ANTs tooling that is used for preprocessing must be built from source,
and the SimpleITK conda package must also be built before installation.

The following steps will build the SimpleITK conda package and place it in
your local conda repository for future install:
```
git clone https://github.com/SimpleITK/SimpleITKCondaRecipe.git
cd SimpleITKCondaRecipe
conda build --python 3.6 recipe
```

The following steps will create a conda environment for building ANTs, install
the cmake and gcc tools as conda packages, and then build the ANTs binaries
in the `~/ants_build/bin/ants/bin/` directory:
```
conda create -n my_build_env python=3.6
conda activate my_build_env
conda install -y cmake gxx_linux-ppc64le=7
cd ~
mkdir ants_build
cd ants_build
git clone https://github.com/ANTsX/ANTs.git
cd ANTs
git checkout v2.3.1
mkdir -p ~/ants_build/bin/ants
cd ~/ants_build/bin/ants
cmake ~/ants_build/ANTs
make -j 120 ANTS
```

3. Install dependencies:
```
conda install pytables lxml scikit-image scikit-learn scipy
pip install nibabel nilearn nipype
conda install --use-local simpleitk
```
(nipype is required for preprocessing only)

4. Add the location of the ANTs binaries to the PATH environmental variable.
If you build the dependencies as described above this will be `~/ants_build/bin/ants/bin/`

5. Add the repository directory to the ```PYTHONPATH``` system variable:
```
cd 3DUNetCNN
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```
6. Convert the data to nifti format and perform image wise normalization and correction:

cd into the brats subdirectory:
```
$ cd brats
```
Import the conversion function and run the preprocessing:
```
$ python
>>> from preprocess import convert_brats_data
>>> convert_brats_data("data/original", "data/preprocessed")
```
Note: By default the preprocessing will process 120
subjects at a time. You can modify the thread count variable
`NUM_FOLDER_PROCESS_THREADS` in preprocess.py to change the concurrency.

7. Run the training:

To run training using the original UNet model (Not LMS enabled):
```
$ python train.py
```

To run training using an improved UNet model (recommended):
```
$ python train_isensee2017.py
```

### Write prediction images from the validation data
In the training above, part of the data was held out for validation purposes.
To write the predicted label maps to file:
```
$ python predict.py
```
The predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for
comparison.

If you have trained the isensee2017 model with the default parameters, the
model name will be generated with a random name. You will need to copy or rename
the model and validation ID files to the file names predict.py expects:
```
$ cp isensee_2017_model.h5 tumor_segmentation_model.h5
$ cp isensee_validation_ids.pkl validation_ids.pkl
```

### Write loss graph and validation score box plots
To create the loss graph and validation score box plot png files run:
```
$ python evaluate.py
```
Note that this uses training.log for the loss graph information. If you
have run the model training multiple times you will need to modify the
training.log file so that it only contains the latest run data. Alternatively,
could could remove the training.log file between runs.

### Results from patch-wise training using original UNet
![Patchwise training loss graph
](doc/brats_64cubedpatch_loss_graph.png)
![Patchwise boxplot scores
](doc/brats_64cubedpatch_validation_scores_boxplot.png)

In the box plot above, the 'whole tumor' area is any labeled area. The 'tumor core' area corresponds to the combination
of labels 1 and 4. The 'enhancing tumor' area corresponds to the 4 label. This is how the BRATS competition is scored.
The both the loss graph and the box plot were created by running the
[evaluate.py](brats/evaluate.py) script in the 'brats'
folder after training has been completed.

### Results from Isensee et al. 2017 model
I (ellisdg) also trained a [model](unet3d/model/isensee2017.py) with the architecture as described in the [2017 BRATS proceedings
](https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf)
on page 100. This [architecture](doc/isensee2017.png) employs a number of changes to the basic UNet including an
[equally weighted dice coefficient](unet3d/metrics.py#L17),
[residual weights](https://wiki.tum.de/display/lfdv/Deep+Residual+Networks),
and [deep supervision](https://arxiv.org/pdf/1409.5185.pdf).
This network was trained using the whole images rather than patches.
As the results below show, this network performed much better than the original UNet.

![Isensee training loss graph
](doc/isensee_2017_loss_graph.png)
![Isensee boxplot scores
](doc/isensee_2017_scores_boxplot.png)

## TensorFlow Large Model Support
### TensorFlow Builds
The TensorFlow Large Model Support integration is written assuming the use of
the TensorFlow build included in IBM Watson Machine Learning Community Edition / IBM PowerAI.

### TensorFlow Large Model Support tuning
You can modify the TensorFlow Large Model Support (TFLMS) tuning by passing command line
parameters. See the training usage for more information:
```
python train_isensee2017.py --help
```

An example command to run the 320^3 size with TFLMS (possible on a 32GB GPU) is:
```
# TF_CUDA_HOST_MEM_LIMIT_IN_MB in TensorFlow < 1.14
# TF_GPU_HOST_MEM_LIMIT_IN_MB in TensorFlow >= 1.14
export TF_CUDA_HOST_MEM_LIMIT_IN_MB=300000
export TF_GPU_HOST_MEM_LIMIT_IN_MB=$TF_CUDA_HOST_MEM_LIMIT_IN_MB

numactl --cpunodebind=0 --membind=0 python train_isensee2017.py --lms --data_file_path=320_data.h5 --image_size 320
```

## Using this code on other 3D datasets
If you want to train a 3D UNet on a different set of data, you can copy either the [train.py](brats/train.py) or the
[train_isensee2017.py](brats/train_isensee2017.py) scripts and modify them to
read in your data rather than the preprocessed BRATS data that they are currently setup to train on.

## Pre-trained Models
The following Keras models were trained on the BRATS 2017 data:
* Isensee et al. 2017:
[model](https://univnebrmedcntr-my.sharepoint.com/:u:/g/personal/david_ellis_unmc_edu/EfSLuSnktwZLs2kB84S8Y6oBRCOk4WT38UxeE9KYka2Gjg)
([weights only](https://univnebrmedcntr-my.sharepoint.com/:u:/g/personal/david_ellis_unmc_edu/EV8SBkKY67xEkk-1o1wiuG8BO-mBxKmd2Pnegvf6St8-DA?e=tRcO71))
* Original U-Net:
[model](https://univnebrmedcntr-my.sharepoint.com/:u:/g/personal/david_ellis_unmc_edu/EUKI2FjEF4FMttJ9q7bQ5IIBEYj7MCJ1O1PF-uTVIV6-YA?e=d2yrEc)
([weights only](https://univnebrmedcntr-my.sharepoint.com/:u:/g/personal/david_ellis_unmc_edu/ESHW544cGtNFlFBKqCY8qHkB79EMRENAyqgQXGIMVMykCQ?e=QLJl5d))

## Citations
GBM Data Citation:
 * Spyridon Bakas, Hamed Akbari, Aristeidis Sotiras, Michel Bilello, Martin Rozycki, Justin Kirby, John Freymann, Keyvan Farahani, and Christos Davatzikos. (2017) Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2017.KLXWJJ1Q

LGG Data Citation:
 * Spyridon Bakas, Hamed Akbari, Aristeidis Sotiras, Michel Bilello, Martin Rozycki, Justin Kirby, John Freymann, Keyvan Farahani, and Christos Davatzikos. (2017) Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2017.GJQ7R0EF
