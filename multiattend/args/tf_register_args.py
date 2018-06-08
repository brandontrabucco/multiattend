###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################

from multiattend.args.tf_static_args import TFStaticArgs

class TFRegisterArgs(object):

    def __init__(
            self):
        pass

    def __call__(
            self, 
            sname, 
            stype, 
            sdefault):
        TFStaticArgs.add_static_arg(
            sname, 
            stype, 
            sdefault)

    def parse_args(self):
        return TFStaticArgs.parse_all_args()