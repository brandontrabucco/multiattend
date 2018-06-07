###########################
# Reset working directory #
###########################
import os
os.chdir("/home/brand/Research/multifaceted_attention")
###########################
# MultiAttend Package.... #
###########################
from multiattend.model.tf_model import TFModel
from multiattend.dataset.tf_dataset import TFDataset

class TFExperiment(object):

    def __init__(
            self):
        self.model = TFModel()
        self.dataset = TFDataset()

    def train(
            self):
        return self.model.train(
            self.dataset.get_training_batch,
            100)