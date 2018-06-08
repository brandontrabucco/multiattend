###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
from multiattend.dataset.io_synthesis import TRAIN_EXAMPLES
from multiattend.dataset.io_synthesis import VAL_EXAMPLES
from multiattend.dataset.io_synthesis import DATASET_COLUMNS
from multiattend.dataset.io_synthesis import DATASET_RANGE
DATASET_FILENAMES = {
    "train": ["multiattend/dataset/csv/train_dataset.csv"],
    "val": ["multiattend/dataset/csv/val_dataset.csv"]}
MIN_RATIO = 10
DATASET_DEFAULT = DATASET_RANGE // 2
BATCH_SIZE = 32
TRAIN_EPOCH_SIZE = TRAIN_EXAMPLES // BATCH_SIZE
VAL_EPOCH_SIZE = VAL_EXAMPLES // BATCH_SIZE
NUM_THREADS = 2