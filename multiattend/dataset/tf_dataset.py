###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
from multiattend.dataset.tf_dataset_args import TFDatasetArgs
from multiattend.dataset.tf_dataset_utils import TFDatasetUtils
import tensorflow as tf

class TFDataset(object):

    def __init__(
            self,
            parser=None):
        self.tf_dataset_args = TFDatasetArgs()
        self.tf_dataset_utils = TFDatasetUtils()

    def get_training_batch(
            self):
        args = self.tf_dataset_args()
        self.tf_dataset_utils.set_args(args)
        queue = tf.train.string_input_producer(
            args.train_dataset)
        token_inputs, token_labels = self.tf_dataset_utils.decode_record(
            queue)
        inputs_batch, labels_batch = self.tf_dataset_utils.generate_batch(
            token_inputs,
            token_labels,
            args.train_instances,
            args.train_instances // args.min_ratio)
        return (inputs_batch, labels_batch)

    def get_val_batch(
            self):
        args = self.tf_dataset_args()
        self.tf_dataset_utils.set_args(args)
        queue = tf.train.string_input_producer(
            args.val_dataset)
        token_inputs, token_labels = self.tf_dataset_utils.decode_record(
            queue)
        inputs_batch, labels_batch = self.tf_dataset_utils.generate_batch(
            token_inputs,
            token_labels,
            args.val_instances,
            args.val_instances // args.min_ratio)
        return (inputs_batch, labels_batch)

    def __call__(
            self):
        train_batch = self.get_training_batch()
        val_batch = self.get_val_batch()
        return train_batch, val_batch