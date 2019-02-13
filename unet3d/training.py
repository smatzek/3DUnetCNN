import math
from functools import partial

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.models import load_model

from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)
from tensorflow.python.keras.callbacks import Callback
import ctypes
_cudart = ctypes.CDLL('libcudart.so')

from tensorflow_large_model_support import LMS
# Set tf logging to INFO for LMS messages
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import sys
import os

# The distribution module
dist_mod = None
if "DDL_OPTIONS" in os.environ:
  import ddl
  dist_mod = ddl

if "USE_HOROVOD_3DUNET" in os.environ:
  import horovod.keras as hvd
  dist_mod = hvd
# In newer versions of Keras this is now set in ~/.keras/keras.json as:
# "image_dim_ordering": "tf"
# K.set_image_dim_ordering('th')

dist_mech = None
if 'horovod' in sys.modules:
    dist_mech = 'horovod'
elif 'ddl' in sys.modules:
    dist_mech = 'ddl'

# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None, lms=False,
                  swapout_threshold=-1,
                  swapin_groupby=-1,
                  swapin_ahead=-1,
                  serialization=-1,
                  sync_mode=0,
                  cuda_profile_epoch=0, cuda_profile_batch_start=0,
                  cuda_profile_batch_end=0):
    callbacks = list()
    if dist_mech is 'ddl':
      callbacks.append(ddl.DDLCallback())
    if not dist_mod or dist_mod.rank() == 0:
      callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
      callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))

    if lms:
        serialization_list = []
        if serialization > 0:
            serialization_list.append('%s:' % serialization)

        lms_callback = LMS(swapout_threshold=swapout_threshold,
                           swapin_groupby=swapin_groupby,
                           swapin_ahead=swapin_ahead,
                           sync_mode=sync_mode,
                           serialization=serialization_list)
        lms_callback.batch_size = 1
        callbacks.append(lms_callback)
    if dist_mech is 'ddl':
      callbacks.append(ddl.DDLGlobalVariablesCallback())
    elif dist_mech is 'horovod':
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      #print("************horovod callback added***************")
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    if cuda_profile_epoch:
        callbacks.append(CudaProfileCallback(cuda_profile_epoch,
                                             cuda_profile_batch_start,
                                             cuda_profile_batch_end))

    return callbacks


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    try:
        from unet3d.model.instancenorm import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error


def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None,
                lms=False,
                swapout_threshold=-1,
                swapin_groupby=-1,
                swapin_ahead=-1,
                serialization=-1,
                sync_mode=0,
                cuda_profile_epoch=0,
                cuda_profile_batch_start=0,
                cuda_profile_batch_end=0):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return:
    """
    model.fit_generator(generator=training_generator,
                        verbose=1 if not dist_mod or dist_mod.rank() == 0 else 0,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=get_callbacks(model_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience,
                                                lms=lms,
                                                swapout_threshold=swapout_threshold,
                                                swapin_groupby=swapin_groupby,
                                                swapin_ahead=swapin_ahead,
                                                serialization=serialization,
                                                sync_mode=sync_mode,
                                                cuda_profile_epoch=cuda_profile_epoch,
                                                cuda_profile_batch_start=cuda_profile_batch_start,
                                                cuda_profile_batch_end=cuda_profile_batch_end))

class CudaProfileCallback(Callback):
    def __init__(self, profile_epoch, profile_batch_start, profile_batch_end):
        self._epoch = profile_epoch - 1
        self._start = profile_batch_start
        self._end = profile_batch_end
        self.epoch_keeper = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_keeper = epoch
    def on_batch_begin(self, batch, logs=None):
        if batch == self._start and self.epoch_keeper == self._epoch:
            print('Starting cuda profiler')
            ret = _cudart.cudaProfilerStart()
            if ret != 0:
              raise Exception("cudaProfilerStart() returned %d" % ret)
        if batch == self._end and self.epoch_keeper == self._epoch:
            print('Stopping cuda profiler')
            ret = _cudart.cudaProfilerStop()
            if ret != 0:
              raise Exception("cudaProfilerStop() returned %d" % ret)
