import sys
import numpy as np

def parseDhParamFile(dhConfigFile):
    assert(dhConfigFile is not None)
    lineContents = None
    with open(dhConfigFile, "r") as f:
        lineContents = f.readlines()

    assert(f.closed)
    assert(lineContents is not None)
    # maybe not the most efficient/clean/etc. way to do this, but should only have to be done once so NBD
    dhParams = np.asarray([line.rstrip().split(',') for line in lineContents[1:]])
    dhParams = dhParams.astype(float)
    return dhParams


### TODO: parse a pox parameter file
def parse_pox_param_file(dhConfigFile):
    pass