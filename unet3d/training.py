import time
import csv
import math
from functools import partial

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model

from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)
from tensorflow.keras.callbacks import Callback
import ctypes
_cudart = ctypes.CDLL('libcudart.so')

#from tensorflow_large_model_support import LMS
# Set tf logging to INFO for LMS messages
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.INFO)
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
                  early_stopping_patience=None, callbacks_config=dict()):
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

    if dist_mech is 'ddl':
      callbacks.append(ddl.DDLGlobalVariablesCallback())
    elif dist_mech is 'horovod':
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      #print("************horovod callback added***************")
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    if callbacks_config.get('cuda_profile_epoch'):
        callbacks.append(CudaProfileCallback(callbacks_config['cuda_profile_epoch'],
                                             callbacks_config['cuda_profile_batch_start'],
                                             callbacks_config['cuda_profile_batch_end']))
    callbacks.append(LMSStats(callbacks_config['lms_stats_logfile']))

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
                learning_rate_patience=20, early_stopping_patience=None, callbacks_config=dict()):
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
                                                logging_file=callbacks_config['training_log_file'],
                                                callbacks_config=callbacks_config))

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

class LMSStats(Callback):
    def __init__(self, logfile):
        self._epoch=0
        self._logfile = logfile

    def set_params(self, params):
        with open(self._logfile, 'w', newline='') as csvfile:
            statswriter = csv.writer(csvfile)
            statswriter.writerow(['step type', 'epoch', 'step', 
                                  'duration', 'allocs', 'reclaimOnes',
                                  'reclaimAlls', #'defrags',
                                  'GiB reclaimed',])# 'GiB defragged'])

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        self.get_start_numbers()

    def on_test_batch_begin(self, batch, logs=None):
        self.get_start_numbers()

    def on_train_batch_end(self, batch, logs=None):
        self.log_end('t', batch)

    def on_test_batch_end(self, batch, logs=None):
        self.log_end('v', batch)
    
    def get_start_numbers(self):
        self._batch_start = time.time()
        self._num_reclaims = tf.experimental.get_num_single_reclaims(0)
        self._num_reclaimAll = tf.experimental.get_num_full_reclaims(0)
#        self._defrags = tf.experimental.get_num_defragmentations(0)
        self._bytes_reclaimed = tf.experimental.get_bytes_reclaimed(0)
        self._num_allocs = tf.experimental.get_num_allocs(0)
#        self._bytes_defragged = tf.experimental.get_bytes_defragged(0)
     
    def log_end(self, step_type, batch_num):
        row = [step_type, self._epoch, batch_num]
        row.append(time.time()-self._batch_start) # duration
        row.append(tf.experimental.get_num_allocs(0)-self._num_allocs) # allocs
        row.append(tf.experimental.get_num_single_reclaims(0)-self._num_reclaims) # reclaims
        row.append(tf.experimental.get_num_full_reclaims(0)-self._num_reclaimAll) # reclaimAlls
#        row.append(tf.experimental.get_num_defragmentations(0) - self._defrags) # defrags
        row.append(((tf.experimental.get_bytes_reclaimed(0)-self._bytes_reclaimed) / 1073741824.0)) # GiB reclaimed
#        row.append(((tf.experimental.get_bytes_defragged(0)-self._bytes_defragged) / 1073741824.0)) # GiB defragged
        
        with open(self._logfile, 'a+', newline='') as csvfile:
            statswriter = csv.writer(csvfile)
            statswriter.writerow(row)
