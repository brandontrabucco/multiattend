###########################
# Reset working directory #
###########################
import os
os.chdir("/home/brand/Research/multifaceted_attention")
###########################
# MultiAttend Package.... #
###########################
from multiattend.args.tf_register_args import TFRegisterArgs

class TFModelArgs(object):

    def __init__(
            self):
        self.register = TFRegisterArgs()

    def __call__(
            self):
        return self.register.parse_args()