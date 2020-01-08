import numpy as np

import utils

# Is this really necessary ??

# Returned array is int
def read(filename):
    return np.int8( utils.read_array(filename) )

def write(labelmap, filename):
    utils.write_array(labelmap, filename)

