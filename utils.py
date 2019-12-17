import numpy as np
import h5py

import mrcfile
import warnings
warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos with mrcfile

from skimage.measure import block_reduce
from lxml import etree
from copy import deepcopy
from sklearn.metrics import pairwise_distances

import matplotlib
matplotlib.use('agg') # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'
import matplotlib.pyplot as plt

#def bin_data:

# Realizes quick visualization of a volume, by plotting its orthoslices, in the same fashion as the matlab function 'tom_volxyz' (TOM toolbox) 
# If volume type is int8, the function assumes that the volume is a labelmap, and hence plots in color scale.
# Else, it assumes that the volume is tomographic data, and plots in gray scale.
def plot_volume_orthoslices(vol, filename):
    # Get central slices along each dimension:
    dim = vol.shape
    idx0 = np.round(dim[0]/2)
    idx1 = np.round(dim[1]/2)
    idx2 = np.round(dim[2]/2)

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

def write_objlist(objlist, filename):
    tree = etree.ElementTree(objlist)
    tree.write(filename, pretty_print=True)
    
def read_objlist(filename):
    tree = etree.parse(filename)
    objlist = tree.getroot()
    return objlist
    
def print_objlist(objlist):
    print(etree.tostring(objlist))

# /!\ for now this function does not know how to handle empty objlists
def get_Ntp_from_objlist(objl_gt, objl_df, tol_pos_err):
    # tolerated position error (in voxel)
    Ngt = len(objl_gt)
    Ndf = len(objl_df)
    coords_gt = np.zeros((Ngt,3))
    coords_df = np.zeros((Ndf,3))

    for idx in range(0,Ngt):
        coords_gt[idx,0] = objl_gt[idx].get('x')
        coords_gt[idx,1] = objl_gt[idx].get('y')
        coords_gt[idx,2] = objl_gt[idx].get('z')
    for idx in range(0,Ndf):
        coords_df[idx,0] = objl_df[idx].get('x')
        coords_df[idx,1] = objl_df[idx].get('y')
        coords_df[idx,2] = objl_df[idx].get('z')

    # Get pairwise distance matrix:
    D = pairwise_distances(coords_gt, coords_df, metric='euclidean')

    # Get pairs that are closer than tol_pos_err:
    D = D<=tol_pos_err

    # A detected object is considered a true positive (TP) if it is closer than tol_pos_err to a ground truth object.
    match_vector = np.sum(D,axis=1)
    Ntp = np.sum(match_vector==1)
    return Ntp
    
def objlist_get_class(objlistIN, label):
    N = len(objlistIN)
    label_list = np.zeros((N,))
    for idx in range(0,N):
        label_list[idx] = objlistIN[idx].get('class_label')
    idx_class = np.nonzero(label_list==label)
    idx_class = idx_class[0]
    
    objlistOUT = etree.Element('objlist')
    for idx in range(0,len(idx_class)):
        objlistOUT.append( deepcopy(objlistIN[idx_class[idx]]) ) # deepcopy is necessary, else the object is removed from objlIN when appended to objlOUT
    return objlistOUT
    
def objlist_above_thr(objlistIN, thr):
    N = len(objlistIN)
    clust_size_list = np.zeros((N,))
    for idx in range(0,N):
        clust_size_list[idx] = objlistIN[idx].get('cluster_size')
    idx_thr = np.nonzero(clust_size_list>=thr)
    idx_thr = idx_thr[0]
    
    objlistOUT = etree.Element('objlist')
    for idx in range(0,len(idx_thr)):
        objlistOUT.append( deepcopy(objlistIN[idx_thr[idx]]) ) # deepcopy is necessary, else the object is removed from objlIN when appended to objlOUT
    return objlistOUT
    
def objlist_scale_coord(objlist, scale):
    for p in range(0,len(objlist)):
        x = int(np.round(float( objlist[p].get('x') )))
        y = int(np.round(float( objlist[p].get('y') )))
        z = int(np.round(float( objlist[p].get('z') )))
        x = scale*x
        y = scale*y
        z = scale*z
        objlist[p].set('x', str(x))
        objlist[p].set('y', str(y))
        objlist[p].set('z', str(z))
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
    
def read_h5array(filename): # rename to read_
    h5file    = h5py.File(filename, 'r')
    dataArray = h5file['dataset'][:]
    h5file.close()
    return dataArray
    
def write_h5array(array, filename):
    h5file = h5py.File(filename, 'w')
    dset     = h5file.create_dataset('dataset', array.shape, dtype='float16' )
    dset[:]  = np.float16(array)
    h5file.close()
    
def load_scoremaps(filename): # rename to read_
    h5file = h5py.File(filename, 'r')
    datasetnames = h5file.keys()
    Ncl = len(datasetnames)
    dim = h5file['class0'].shape 
    scoremaps = np.zeros((dim[0],dim[1],dim[2],Ncl))
    for cl in range(0,Ncl):
        scoremaps[:,:,:,cl] = h5file['class'+str(cl)][:]
    h5file.close()
    return scoremaps
    
def write_scoremaps(scoremaps, filename):
    h5file = h5py.File(filename, 'w')
    dim = scoremaps.shape
    Ncl = dim[3]
    for cl in range(0,Ncl):
	    dset = h5file.create_dataset('class'+str(cl), (dim[0], dim[1], dim[2]), dtype='float16' )
	    dset[:] = np.float16(scoremaps[:,:,:,cl])
    h5file.close()
    
def write_labelmap(labelmap, filename):
    dim = labelmap.shape
    h5file = h5py.File(filename, 'w')
    dset     = h5file.create_dataset('dataset', (dim[0],dim[1],dim[2]), dtype='int8' )
    dset[:]  = np.int8(labelmap)
    h5file.close()

def read_mrc(filename):
    with mrcfile.open(filename, permissive=True) as mrc:
        array = mrc.data
    return array

def write_mrc(array, filename):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(array)

