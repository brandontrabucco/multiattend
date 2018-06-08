###########################
# Reset working directory #
###########################
import os
os.chdir("/home/btrabucco/research/multiattend")
###########################
# MultiAttend Package.... #
###########################

from collections import namedtuple
import argparse

class TFStaticArgs(object):

    StaticArg = namedtuple("StaticArg", "sname stype sdefault")
    static_args = []
    parsed_args = None

    @staticmethod
    def add_static_arg(
            sname, 
            stype,
            sdefault):
        TFStaticArgs.static_args += [TFStaticArgs.StaticArg(
            sname=sname, 
            stype=stype, 
            sdefault=sdefault)]

    @staticmethod
    def parse_all_args():
        if TFStaticArgs.parsed_args is None:
            parser = argparse.ArgumentParser(
                    description="TFStaticArgs")
            for a in TFStaticArgs.static_args:
                parser.add_argument(
                    a.sname,
                    type=a.stype,
                    default=a.sdefault)
            TFStaticArgs.parsed_args = parser.parse_args()
        return TFStaticArgs.parsed_args