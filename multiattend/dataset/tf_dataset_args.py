###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
from multiattend.args.tf_register_args import TFRegisterArgs
from multiattend.dataset import TRAIN_EXAMPLES
from multiattend.dataset import VAL_EXAMPLES
from multiattend.dataset import DATASET_COLUMNS
from multiattend.dataset import DATASET_RANGE
from multiattend.dataset import DATASET_FILENAMES
from multiattend.dataset import MIN_RATIO
from multiattend.dataset import DATASET_DEFAULT
from multiattend.dataset import BATCH_SIZE
from multiattend.dataset import TRAIN_EPOCH_SIZE
from multiattend.dataset import VAL_EPOCH_SIZE
from multiattend.dataset import NUM_THREADS
import argparse

class TFDatasetArgs(object):

    def __init__(
            self):
        self.register = TFRegisterArgs()
        self.register("--train_dataset", str, DATASET_FILENAMES["train"])
        self.register("--val_dataset", str, DATASET_FILENAMES["val"])
        self.register("--train_instances", int, TRAIN_EXAMPLES)
        self.register("--val_instances", int, VAL_EXAMPLES)
        self.register("--min_ratio", float, MIN_RATIO)
        self.register("--dataset_columns", int, DATASET_COLUMNS)
        self.register("--dataset_range", int, DATASET_RANGE)
        self.register("--dataset_default", int, DATASET_DEFAULT)
        self.register("--batch_size", int, BATCH_SIZE)
        self.register("--train_epoch_size", int, TRAIN_EPOCH_SIZE)
        self.register("--val_epoch_size", int, VAL_EPOCH_SIZE)
        self.register("--num_threads", int, NUM_THREADS)

    def __call__(
            self):
        return self.register.parse_args()
