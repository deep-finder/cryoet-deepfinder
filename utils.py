import os
import numpy as np
import h5py

import mrcfile
import warnings
warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos with mrcfile

from skimage.measure import block_reduce

import matplotlib
matplotlib.use('agg') # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'
import matplotlib.pyplot as plt

# Realizes quick visualization of a volume, by plotting its orthoslices, in the same fashion as the matlab function 'tom_volxyz' (TOM toolbox) 
# If volume type is int8, the function assumes that the volume is a labelmap, and hence plots in color scale.
# Else, it assumes that the volume is tomographic data, and plots in gray scale.
def plot_volume_orthoslices(vol, filename):
    # Get central slices along each dimension:
    dim = vol.shape
    idx0 = np.int( np.round(dim[0]/2) )
    idx1 = np.int( np.round(dim[1]/2) )
    idx2 = np.int( np.round(dim[2]/2) )

    slice0 = vol[idx0,:,:]
    slice1 = vol[:,idx1,:]
    slice2 = vol[:,:,idx2]

    # Build image containing orthoslices:
    img_array = np.zeros((slice0.shape[0]+slice1.shape[0], slice0.shape[1]+slice1.shape[0]))
    img_array[0:slice0.shape[0], 0:slice0.shape[1]] = slice0
    img_array[slice0.shape[0]-1:-1, 0:slice0.shape[1]] = slice1
    img_array[0:slice0.shape[0], slice0.shape[1]-1:-1] = np.flipud(np.rot90(slice2))

    # Drop the plot:
    fig = plt.figure(figsize=(10,10))
    if vol.dtype==np.int8:
        plt.imshow(img_array, cmap='CMRmap', vmin=np.min(vol), vmax=np.max(vol))
    else:
        mu  = np.mean(vol) # Get mean and std of data for plot range:
        sig = np.std(vol)
        plt.imshow(img_array, cmap='gray', vmin=mu-5*sig, vmax=mu+5*sig)
    fig.savefig(filename)

def read_h5array(filename):
    h5file = h5py.File(filename, 'r')
    dataArray = h5file['dataset'][:]
    h5file.close()
    return dataArray

def write_h5array(array, filename):
    h5file = h5py.File(filename, 'w')
    if array.dtype == np.int8:
        dset = h5file.create_dataset('dataset', array.shape, dtype='int8')
        dset[:] = np.int8(array)
    else:
        dset = h5file.create_dataset('dataset', array.shape, dtype='float16')
        dset[:] = np.float16(array)
    h5file.close()

def read_mrc(filename):
    with mrcfile.open(filename, permissive=True) as mrc:
        array = mrc.data
    return array

def write_mrc(array, filename):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(array)

def read_array(filename):
    data_format = os.path.splitext(filename)
    if data_format[1] == '.h5':
        array = read_h5array(filename)
    elif data_format[1] == '.mrc':
        array = read_mrc(filename)
    else:
        print('/!\ DeepFinder can only read datasets in .h5 and .mrc formats')
    return array

def write_array(array, filename):
    data_format = os.path.splitext(filename)
    if data_format[1] == '.h5':
        write_h5array(array, filename)
    elif data_format[1] == '.mrc':
        write_mrc(array, filename)
    else:
        print('/!\ DeepFinder can only write arrays in .h5 and .mrc formats')

def bin_array(array):
    return block_reduce(array, (2,2,2), np.mean)
