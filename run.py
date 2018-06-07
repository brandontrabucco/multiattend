###########################
# Reset working directory #
###########################
import os
os.chdir("/home/brand/Research/multifaceted_attention")
###########################
# MultiAttend Package.... #
###########################
from multiattend.tf_experiment import TFExperiment

if __name__ == "__main__":
    experiment = TFExperiment()
    experiment.train()