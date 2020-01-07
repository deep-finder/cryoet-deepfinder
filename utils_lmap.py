import numpy as np

import utils

# Is this really necessary ??

# Returned array is int
def read(filename):
    return np.int8( utils.read_array(filename) )

def write(labelmap, filename):
    utils.write_array(labelmap, filename)

# # Written array is int instead of float (takes less space!)
# def write_h5(labelmap, filename):
#     dim = labelmap.shape
#     h5file = h5py.File(filename, 'w')
#     dset     = h5file.create_dataset('dataset', (dim[0],dim[1],dim[2]), dtype='int8' )
#     dset[:]  = np.int8(labelmap)
#     h5file.close()
#
# def write(array, filename):
#     data_format = os.path.splitext(filename)
#     if data_format[1] == '.h5':
#         write_h5(array, filename)
#     elif data_format[1] == '.mrc':
#         utils.write_mrc(np.int8(array), filename)
#     else:
#         print('/!\ DeepFinder can only write arrays in .h5 and .mrc formats')
