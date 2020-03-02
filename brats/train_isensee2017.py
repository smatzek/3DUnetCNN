import argparse
import os
import glob
import random
import string
import tensorflow as tf
# import Horovod if invoked with distribution
hvd = None
if "OMPI_COMM_WORLD_RANK" in os.environ:
    import horovod.tensorflow.keras as hvd
    hvd.init()

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model


FLAGS = None

def setup_input_shape():
    if "patch_shape" in config and config["patch_shape"] is not None:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
    else:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))


config = dict()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = (1, 2, 4)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1Gd", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
setup_input_shape()
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
if hvd:
    config["initial_learning_rate"] = 5e-4 * hvd.size()
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("brats_data.h5")
config["model_file"] = os.path.abspath("isensee_2017_model.h5")
config["training_file"] = os.path.abspath("isensee_training_ids.pkl")
config["validation_file"] = os.path.abspath("isensee_validation_ids.pkl")
config["training_log_file"] = 'training.csv'
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
config['lms_stats_logfile'] = 'lms_stats'
config['lms_average_stats_logfile'] = 'lms_average_stats'


def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False):

    if FLAGS.lms:
        tf.config.experimental.set_lms_enabled(True)
        print('LMS Enabled')
        if FLAGS.lms_defrag:
            tf.config.experimental.set_lms_defrag_enabled(True)
            print('LMS Defragmentation Enabled')
        else:
            print('LMS Defragmentation Disabled')

    else:
        print('LMS Disabled')

    if hvd:
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)
        tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if FLAGS.gpu_memory_limit:
        print('Limiting GPU memory to %s MB' % FLAGS.gpu_memory_limit)
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(
          physical_devices[0],
          [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.gpu_memory_limit)])

    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids)
    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = isensee2017_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"])

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    if FLAGS.steps_per_epoch:
        n_train_steps = FLAGS.steps_per_epoch
    if FLAGS.validation_steps:
        n_validation_steps = FLAGS.validation_steps

    callbacks_config = {'cuda_profile_epoch': FLAGS.cuda_profile_epoch,
                        'cuda_profile_batch_start': FLAGS.cuda_profile_batch_start,
                        'cuda_profile_batch_end': FLAGS.cuda_profile_batch_end,
                        'training_log_file': config["training_log_file"],
                        'lms_stats_logfile': config['lms_stats_logfile'],
                        'lms_average_stats_logfile': config['lms_average_stats_logfile']}
    if FLAGS.lms_stats:
        callbacks_config['lms_stats_enabled'] = True
    if FLAGS.lms_stats_average:
        callbacks_config['lms_stats_average_enabled'] = True
        callbacks_config['image_size'] = FLAGS.image_size
        callbacks_config['batch_size'] = config["batch_size"]
        callbacks_config['lms_stats_warmup_steps'] = FLAGS.lms_stats_warmup_steps

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"],
                callbacks_config=callbacks_config)
    data_file_opened.close()


def generate_stats_name(random_part, root):
    # Generates the name of the output stats file.
    # If Horovod distribution is enabled, the node and GPU ID
    # are appended to the end
    if random_part:
        random_part = '%s_' % random_part
    return ('%s%s%s%s.csv' %
           (random_part, root,
           ('_%s' % os.environ["HOSTNAME"] if hvd else ""),
           (('_gpu%s' % hvd.local_rank()) if hvd else "")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int,
                      default=500,
                      help='Number of epochs to run. (Early stopping still '
                           'applies.) This parameter is useful for measuring '
                           'epoch times by running a few epochs rather than '
                           'a full training run to convergence.')
    parser.add_argument('--image_size', type=int,
                      default=144,
                      help='One dimension of the cubic size of the image. For '
                           'example for 192^3, pass 192.')
    parser.add_argument('--data_file_path', type=str,
                      default='brats_data.h5',
                      help='Path to the h5 data file containing training and '
                           'validation subjects.')
    # LMS parameters
    lms_group = parser.add_mutually_exclusive_group(required=False)
    lms_group.add_argument('--lms', dest='lms', action='store_true',
                           help='Enable TFLMS')
    lms_group.add_argument('--no-lms', dest='lms', action='store_false',
                           help='Disable TFLMS (Default)')
    parser.set_defaults(lms=False)
    parser.add_argument("--gpu_memory_limit", type=int, default=0,
                        help='Set up a single virtual GPU device with the '
                             'specified amount of GPU memory (in MB). '
                             'Disabled by default.')
    defrag_group = parser.add_mutually_exclusive_group(required=False)
    defrag_group.add_argument('--lms_defrag', dest='lms_defrag',
                              action='store_true',
                              help='Enable LMS defragmentation')
    defrag_group.add_argument('--no-lms_defrag', dest='lms_defrag',
                              action='store_false',
                              help='Disable LMS defragmentation (Default)')
    parser.set_defaults(lms_defrag=False)
    lms_stats = parser.add_mutually_exclusive_group(required=False)
    lms_stats.add_argument('--lms_stats', dest='lms_stats', action='store_true',
                           help='Log LMS per-step stats to a file named '
                                '*_lms_stats.csv')
    lms_stats.add_argument('--no-lms_stats', dest='lms_stats',
                           action='store_false',
                           help='Disable logging LMS per-step stats (Default)')
    parser.set_defaults(lms_stats=False)

    lms_stats_average = parser.add_mutually_exclusive_group(required=False)
    lms_stats_average.add_argument('--lms_stats_average',
         dest='lms_stats_average',
         action='store_true',
         help='Log LMS average stats to a file named '
              '*_lms_stats_average.csv')
    lms_stats_average.add_argument('--no-lms_stats_average',
        dest='lms_stats_average', action='store_false',
        help='Disable logging LMS average stats (Default)')
    parser.set_defaults(lms_stats_average=False)

    parser.add_argument('--lms_stats_warmup_steps',
                        default=5,
                        help='The number of steps to train before starting '
                             'LMS statistics recording. (Default 5)',
                        type=int)

    parser.add_argument('--steps_per_epoch', type=int,
                      default=0,
                      help='An override for the number of steps to run in an '
                           'epoch. This is useful when performance profiling '
                           'large resolutions to shorten runtimes. The default '
                           'behavior is to use the number of subjects and '
                           'batch size to calculate the correct number of '
                           'steps.')
    parser.add_argument('--validation_steps', type=int,
                      default=0,
                      help='An override for the number of validation steps to '
                           'run in an epoch. This is useful when performance '
                           'profiling large resolutions to shorten runtimes. '
                           'The default is to use the default number of '
                           'validation steps given the training/validation '
                           'subject split.')
    parser.add_argument('--randomize_output_file_names', type=bool,
                      default=True,
                      help='This will generate and a prepend random name '
                           'for training output files. Default is True')
    parser.add_argument('--cuda_profile_epoch', type=int,
                      default=0,
                      help='The epoch in which to start CUDA profiling '
                           '(nvvp). Default is 0 (no profiling)')
    parser.add_argument('--cuda_profile_batch_start', type=int,
                      default=1,
                      help='The batch in which to start CUDA profiling '
                           '(nvvp). Default is 1.')
    parser.add_argument('--cuda_profile_batch_end', type=int,
                      default=2,
                      help='The batch in which to end CUDA profiling '
                           '(nvvp). Default is 2.')
    FLAGS = parser.parse_args()
    config['n_epochs'] = FLAGS.epochs
    config['image_shape'] = (FLAGS.image_size, FLAGS.image_size, FLAGS.image_size)
    setup_input_shape()
    config['data_file'] = FLAGS.data_file_path
    random_part = ''
    if FLAGS.randomize_output_file_names:
        random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        if not hvd or hvd.rank() == 0:
            print('The prefix for output file names is:', random_part)

        # Only hvd rank 0 or single GPU case will write these files.
        config["model_file"] = os.path.abspath("%s_isensee_2017_model.h5" % random_part)
        config["training_log_file"] = "%s_training.csv" % random_part

        if not hvd or hvd.rank() == 0:
            with open("%s_run_params.txt" % random_part,"w") as paramlog:
                paramlog.write(str(FLAGS))
        if not hvd:
            config['lms_stats_logfile'] = generate_stats_name(random_part,
                                                              config['lms_stats_logfile'])

    # We cannot randomize this file name when using Horovod because
    # the random_part which acts as a "runID" will be different in each
    # GPU process.
    if hvd:
        config['lms_stats_logfile'] = generate_stats_name('', config['lms_stats_logfile'])

    # do not use random_part for the average file so multiple separate
    # runs will append to the same stats file for a given GPU
    config['lms_average_stats_logfile'] = generate_stats_name('', config['lms_average_stats_logfile'])


    if hvd and FLAGS.gpu_memory_limit:
        print('Error: This model does not support limiting GPU memory while '
              'running with Horovod.')
        exit(1)
    main(overwrite=config["overwrite"])
