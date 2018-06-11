# 3D U-Net Convolution Neural Network with Keras
![Tumor Segmentation Example](doc/tumor_segmentation_illusatration.gif)
## Background
Originally designed after [this paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) on
volumetric segmentation with a 3D U-Net.
The code was written to be trained using the
[BRATS](http://www.med.upenn.edu/sbia/brats2017.html) data set for brain tumors, but it can
be easily modified to be used in other 3D applications.

## Tutorial using BRATS Data
### Training
1. Download the BRATS 2017 [GBM](https://app.box.com/shared/static/bpqo6uqmqinke5jkyhbik9va2uq8ky01.zip) and
[LGG](https://app.box.com/shared/static/pqkmy3zcvud2qxlx5poe458azb1dzj54.zip) data. Place the unzipped folders in the
```brats/data/original``` folder.
2. Install dependencies:
```
nibabel,
keras,
pytables,
nilearn,
SimpleITK,
nipype
```
(nipype is required for preprocessing only)

3. Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) and add the location of the ANTs
binaries to the PATH environmental variable.

4. Add the repository directory to the ```PYTONPATH``` system variable:
```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```
5. Convert the data to nifti format and perform image wise normalization and correction:

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
Note: This can take a while.  By default the preprocessing will process one
file at a time. You can modify the thread count in preprocess.py to multithread
this. On a server with 32 cores and SMT enabled you could easily set this to
30 or 60 and greatly reduce your preprocessing time.

6. Run the training:

To run training using the original UNet model:
```
$ python train.py
```

To run training using an improved UNet model (recommended):
```
$ python train_isensee2017.py
```
**If you run out of memory during training:** try setting
```config['patch_shape`] = (64, 64, 64)``` for starters.
Also, read the "Configuration" notes at the bottom of this page.

If you are running the train_isensee2017.py model multiple times you should
run:
```
$ rm isensee*
```
between runs to remove the model and train/validation id files.

### Write prediction images from the validation data
In the training above, part of the data was held out for validation purposes.
To write the predicted label maps to file:
```
$ python predict.py
```
The predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for
comparison.

If you have trained the isensee2017 model you will need to copy or rename
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
I also trained a [model](unet3d/model/isensee2017.py) with the architecture as described in the [2017 BRATS proceedings
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

### Configuration
Changing the configuration dictionary in the [train.py](brats/train.py) or the
[train_isensee2017.py](brats/train_isensee2017.py) scripts, makes it easy to test out different model and
training configurations.
I would recommend trying out the Isensee et al. model first and then modifying the parameters until you have satisfactory
results.
If you are running out of memory, try training using ```(64, 64, 64)``` shaped patches.
Reducing the "batch_size" and "validation_batch_size" parameters will also reduce the amount of memory required for
training as smaller batch sizes feed smaller chunks of data to the CNN.
If the batch size is reduced down to 1 and it still you are still running
out of memory, you could also try changing the patch size to ```(32, 32, 32)```.
Keep in mind, though, that a smaller patch sizes may not perform as well as larger patch sizes.

## Large Model Support tuning
You can modify the Large Model Support (LMS) tuning by passing command line
parameters.  For example: python train_isensee2017.py <n_tensors> <lb> <branch_threshold>

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
