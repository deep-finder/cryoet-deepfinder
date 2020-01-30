import numpy as np
import h5py

from . import common as cm

def bin(scoremaps):
    dim = scoremaps.shape
    Ncl = dim[3]
    dimB0 = np.int(np.round(dim[0]/2))
    dimB1 = np.int(np.round(dim[1]/2))
    dimB2 = np.int(np.round(dim[2]/2))
    dimB = (dimB0, dimB1, dimB2, Ncl)
    scoremapsB = np.zeros(dimB)
    for cl in range(0,Ncl):
        scoremapsB[:,:,:,cl] = cm.bin_array(scoremaps[:,:,:,cl])
    return scoremapsB
    
def to_labelmap(scoremaps):
    labelmap = np.int8( np.argmax(scoremaps,3) )
    return labelmap

def read_h5(filename):
    h5file = h5py.File(filename, 'r')
    datasetnames = h5file.keys()
    Ncl = len(datasetnames)
    dim = h5file['class0'].shape 
    scoremaps = np.zeros((dim[0],dim[1],dim[2],Ncl))
    for cl in range(Ncl):
        scoremaps[:,:,:,cl] = h5file['class'+str(cl)][:]
    h5file.close()
    return scoremaps
    
def write_h5(scoremaps, filename):
    h5file = h5py.File(filename, 'w')
    dim = scoremaps.shape
    Ncl = dim[3]
    for cl in range(Ncl):
        dset = h5file.create_dataset('class'+str(cl), (dim[0], dim[1], dim[2]), dtype='float16' )
        dset[:] = np.float16(scoremaps[:,:,:,cl])
    h5file.close()