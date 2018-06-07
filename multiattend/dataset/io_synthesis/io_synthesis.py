###########################
# Reset working directory #
###########################
import os
os.chdir("/home/brand/Research/multifaceted_attention")
###########################
# MultiAttend Package.... #
###########################
from multiattend.dataset.io_synthesis.io_synthesis_args import IOSynthesisArgs
from multiattend.dataset.io_synthesis.io_synthesis_utils import IOSynthesisUtils

class IOSynthesis(object):

    def __init__(
            self):
        self.io_synthesis_args = IOSynthesisArgs()
        self.io_synthesis_utils = IOSynthesisUtils()

    def get_train_dataset(
            self):
        args = self.io_synthesis_args()
        return self.io_synthesis_utils.get_dataset(
            args.range,
            args.length,
            args.train_instances)

    def get_val_dataset(
            self):
        args = self.io_synthesis_args()
        return self.io_synthesis_utils.get_dataset(
            args.range,
            args.length,
            args.val_instances)

    def __call__(
            self):
        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()
        return train_dataset, val_dataset
        