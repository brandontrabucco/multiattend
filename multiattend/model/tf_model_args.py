###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################
from multiattend.args.tf_register_args import TFRegisterArgs

class TFModelArgs(object):

    def __init__(
            self):
        self.register = TFRegisterArgs()
        self.register("--encoder_depth", int, 512)
        self.register("--decoder_depth", int, 256)
        self.register("--learning_rate", float, 1e-2)
        self.register("--decay_steps", int, 1000)
        self.register("--decay_rate", float, 0.9)
        self.register("--weight_penalty", float, 0.01)
        self.register("--keep_probability", float, 0.9)
        self.register("--checkpoint_after_test", bool, True)
        self.register("--checkpoint_after_train", bool, True)
        self.register("--target_checkpoint", str, None)
        self.register("--use_previous", bool, False)

    def __call__(
            self):
        return self.register.parse_args()