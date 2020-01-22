import os
import argparse

from train import config
from unet3d.prediction import run_validation_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_path', type=str,
                      default='brats_data.h5',
                      help='Path to the h5 data file containing training and '
                           'validation subjects.')
    parser.add_argument('--model_file', type=str,
                      default='isensee_2017_model.h5',
                      help='Path to the model file.')
    parser.add_argument('--output_dir', type=str,
                      default='prediction',
                      help='Output directory')
    parser.add_argument('--validation_ids_file', type=str,
                      default='isensee_validation_ids.pkl',
                      help='The pkl file containing the validation IDs.')
    FLAGS = parser.parse_args()
    config['validation_file'] = FLAGS.validation_ids_file
    config['model_file'] = FLAGS.model_file
    config['data_file'] = FLAGS.data_file_path

    prediction_dir = os.path.abspath(FLAGS.output_dir)
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main()
