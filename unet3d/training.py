import time
import csv
import math
import os
from functools import partial
# import Horovod if invoked with distribution
hvd = None
if "OMPI_COMM_WORLD_RANK" in os.environ:
    import horovod.tensorflow.keras as hvd

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model

from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)
from unet3d.callbacks import CudaProfileCallback, LMSStatsLogger, LMSStatsAverage

import sys
import os

# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.csv", verbosity=1,
                  early_stopping_patience=None, callbacks_config=dict()):
    callbacks = list()

    if hvd:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1))

    # Only add model checkpointing and CSVLogger on the rank=0 node when using
    # Horovod
    if not hvd or hvd.rank() == 0:
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

    if callbacks_config.get('cuda_profile_epoch'):
        callbacks.append(CudaProfileCallback(callbacks_config['cuda_profile_epoch'],
                                             callbacks_config['cuda_profile_batch_start'],
                                             callbacks_config['cuda_profile_batch_end']))
    if callbacks_config.get('lms_stats_enabled'):
        callbacks.append(LMSStatsLogger(callbacks_config['lms_stats_logfile']))

    if callbacks_config.get('lms_stats_average_enabled'):
        lms = LMSStatsAverage(callbacks_config['lms_average_stats_logfile'],
                              callbacks_config['image_size'],
                              batch_size=callbacks_config['batch_size'],
                              start_batch=callbacks_config['lms_stats_warmup_steps'],
                              image_dimensions=3)
        callbacks.append(lms)

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
    model.fit(training_generator,
              steps_per_epoch=steps_per_epoch,
              verbose=1 if not hvd or hvd.rank() == 0 else 0,
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
