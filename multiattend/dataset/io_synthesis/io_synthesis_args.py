###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
from multiattend.args.tf_register_args import TFRegisterArgs
from multiattend.dataset.io_synthesis import TRAIN_EXAMPLES
from multiattend.dataset.io_synthesis import VAL_EXAMPLES
from multiattend.dataset.io_synthesis import DATASET_COLUMNS
from multiattend.dataset.io_synthesis import DATASET_RANGE
import argparse

class IOSynthesisArgs(object):

    def __init__(
            self):
        self.register = TFRegisterArgs()
        self.register("--range", int, DATASET_RANGE)
        self.register("--length", int, DATASET_COLUMNS)
        self.register("--train_instances", int, TRAIN_EXAMPLES)
        self.register("--val_instances", int, VAL_EXAMPLES)

    def __call__(
            self):
        return self.register.parse_args()
