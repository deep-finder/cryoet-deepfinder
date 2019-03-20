import numpy as np
import h5py
from skimage.measure import block_reduce
from lxml import etree

#def bin_data:

def write_objlist(objlist, filename):
    tree = etree.ElementTree(objlist)
    tree.write(filename, pretty_print=True)
    
def read_objlist(filename):
    tree = etree.parse(filename)
    objlist = tree.getroot()
    return objlist

def bin_scoremaps(scoremaps):
    dim = scoremaps.shape
    Ncl = dim[3]
    dimB = (np.round(dim[0]/2), np.round(dim[1]/2), np.round(dim[2]/2), Ncl)
    scoremapsB = np.zeros(dimB)
    for cl in range(0,Ncl):
        scoremapsB[:,:,:,cl] = block_reduce(scoremaps[:,:,:,cl], (2,2,2), np.mean)
    return scoremapsB
    
def scoremaps2labelmap(scoremaps):
    labelmap = np.argmax(scoremaps,3)
    return labelmap
    
def load_h5array(filename): # rename to read_
    h5data    = h5py.File(filename, 'r')
    dataArray = h5data['dataset'][:]
    h5data.close()
    return dataArray
    
def load_scoremaps(filename): # rename to read_
    h5data = h5py.File(filename, 'r')
    datasetnames = h5data.keys()
    Ncl = len(datasetnames)
    dim = h5data['class0'].shape 
    scoremaps = np.zeros((dim[0],dim[1],dim[2],Ncl))
    for cl in range(0,Ncl):
        scoremaps[:,:,:,cl] = h5data['class'+str(cl)][:]
    h5data.close()
    return scoremaps
    
def write_scoremaps(scoremaps, filename):
    h5scoremap = h5py.File(filename, 'w')
    dim = scoremaps.shape
    Ncl = dim[3]
    for cl in range(0,Ncl):
	    dset = h5scoremap.create_dataset('class'+str(cl), (dim[0], dim[1], dim[2]), dtype='float16' )
	    dset[:] = np.float16(scoremaps[:,:,:,cl])
    h5scoremap.close()
    
def write_labelmap(labelmap, filename):
    dim = labelmap.shape
    h5lblmap = h5py.File(filename, 'w')
    dset     = h5lblmap.create_dataset('dataset', (dim[0],dim[1],dim[2]), dtype='int8' )
    dset[:]  = np.int8(labelmap)
    h5lblmap.close()