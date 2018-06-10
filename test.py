###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
from multiattend.tf_experiment import TFExperiment

if __name__ == "__main__":
    experiment = TFExperiment()
    experiment.test()