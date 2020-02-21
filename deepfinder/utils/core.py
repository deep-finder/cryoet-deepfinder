# This file contains classes/functions that are judged not necessary for the user.

import numpy as np
import h5py
import os

import matplotlib
matplotlib.use('agg') # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'
import matplotlib.pyplot as plt

from . import common as cm

class DeepFinder:
    def __init__(self):
        self.obs_list = [observer_print]

    # Useful for sending prints to GUI
    def set_observer(self, obs):
        self.obs_list.append(obs)

    # "Master print" calls all observers for prints
    def display(self, message):
        for obs in self.obs_list: obs.display(message)

# Following observer classes are needed to send prints to GUI:
class observer_print:
    def display(message):
        print(message)

class observer_gui:
    def __init__(self, pyqt_signal):
        self.sig = pyqt_signal
    def display(self, message):
        self.sig.emit(message)


# This functions loads the training set at specified paths.
# INPUTS:
#   path_data  : list of strings '/path/to/tomogram.ext'
#   path_target: list of strings '/path/to/target.ext'
#                The idx of above lists correspond to each other so that (path_data[idx], path_target[idx]) corresponds
#                to a (tomog, target) pair
#   dset_name  : can be usefull if files are stored as .h5
# OUTPUTS:
#   data_list  : list of 3D numpy arrays (tomograms)
#   target_list: list of 3D numpy arrays (annotated tomograms)
#                In the same way as for the inputs, (data_list[idx],target_list[idx]) corresponds to a (tomo,target) pair
# TODO: move to common?
def load_dataset(path_data, path_target, dset_name='dataset'):
    data_list   = []
    target_list = []
    for idx in range(0,len(path_data)):
        data_list.append(  cm.read_array(path_data[idx]  , dset_name))
        target_list.append(cm.read_array(path_target[idx], dset_name))
    return data_list, target_list

# This function applies bootstrap (i.e. re-sampling) in case of unbalanced classes.
# Given an objlist containing objects from various classes, this function outputs an equal amount of objects for each
# class, each objects being uniformely sampled inside its class set.
# INPUTS:
#   objlist: list of dictionaries
#   Nbs    : number of objects to sample from each class
# OUTPUT:
#   bs_idx : list of indexes corresponding to the bootstraped objects
def get_bootstrap_idx(objlist,Nbs):
    # Get a vector containing the object class labels (from objlist):
    Nobj = len(objlist)
    label_list = np.zeros((Nobj,))
    for oo in range(0,Nobj):
        label_list[oo] = float( objlist[oo]['label'] )
        
    lblTAB = np.unique(label_list) # vector containing unique class labels 
        
    # Bootstrap data so that we have equal frequencies (1/Nbs) for all classes:
    # ->from label_list, sample Nbs objects from each class
    bs_idx = []
    for l in lblTAB:
        bs_idx.append( np.random.choice(np.squeeze(np.asarray(np.nonzero(label_list==l))), Nbs) ) # TODO: can be simplified
    bs_idx = np.concatenate(bs_idx)
    return bs_idx

# Takes position specified in 'obj', applies random shift to it, and then checks if the patch around this position is
# out of the tomogram boundaries. If so, the position is shifted to that patch is inside the tomo boundaries.
# INPUTS:
#   tomodim: tuple (dimX,dimY,dimZ) containing size of tomogram
#   p_in   : int lenght of patch in voxels
#   obj    : dictionary obtained when calling objlist[idx]
#   Lrnd   : int random shift in voxels applied to position
# OUTPUTS:
#   x,y,z  : int,int,int coordinates for sampling patch safely
def get_patch_position(tomodim, p_in, obj, Lrnd):
    # sample at coordinates specified in obj=objlist[idx]
    x = int( obj['x'] )
    y = int( obj['y'] )
    z = int( obj['z'] )
        
    # Add random shift to coordinates:
    x = x + np.random.choice(range(-Lrnd,Lrnd+1))
    y = y + np.random.choice(range(-Lrnd,Lrnd+1))
    z = z + np.random.choice(range(-Lrnd,Lrnd+1))
    
    # Shift position if too close to border:
    if (x<p_in) : x = p_in
    if (y<p_in) : y = p_in
    if (z<p_in) : z = p_in
    if (x>tomodim[2]-p_in): x = tomodim[2]-p_in
    if (y>tomodim[1]-p_in): y = tomodim[1]-p_in
    if (z>tomodim[0]-p_in): z = tomodim[0]-p_in

    #else: # sample random position in tomogram
    #    x = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    #    y = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    #    z = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    
    return x,y,z

# Saves training history as .h5 file.
# INPUTS:
#   history: dictionary object containing lists. These lists contain scores and metrics wrt epochs.
#   filename: string '/path/to/net_train_history.h5'
def save_history(history, filename):
    if os.path.isfile(filename): # if file exists, delete before writing the updated version
        os.remove(filename)      # quick fix for OSError: Can't write data (no appropriate function for conversion path)

    h5file = h5py.File(filename, 'w')

    # train and val loss & accuracy:
    dset    = h5file.create_dataset('acc', np.array(history['acc']).shape)
    dset[:] = np.array(history['acc'])
    dset    = h5file.create_dataset('loss', np.array(history['loss']).shape)
    dset[:] = np.array(history['loss'])
    dset    = h5file.create_dataset('val_acc', np.array(history['val_acc']).shape)
    dset[:] = np.array(history['val_acc'])
    dset    = h5file.create_dataset('val_loss', np.array(history['val_loss']).shape)
    dset[:] = np.array(history['val_loss'])

    # val precision, recall, F1:
    dset    = h5file.create_dataset('val_f1', np.array(history['val_f1']).shape)
    dset[:] = np.array(history['val_f1'])
    dset    = h5file.create_dataset('val_precision', np.array(history['val_precision']).shape)
    dset[:] = np.array(history['val_precision'])
    dset    = h5file.create_dataset('val_recall', np.array(history['val_recall']).shape)
    dset[:] = np.array(history['val_recall'])

    h5file.close()
    return

# Plots the training history as several graphs and saves them in an image file.
# Validation score is averaged over all batches tested in validation step (steps_per_valid)
# Training score is averaged over last N=steps_per_valid batches of each epoch.
#   -> This is to have similar curve smoothness to validation.
# INPUTS:
#   history: dictionary object containing lists. These lists contain scores and metrics wrt epochs.
#   filename: string '/path/to/net_train_history_plot.png'
def plot_history(history, filename):
    Ncl = len(history['val_f1'][0])
    legend_names = []
    for lbl in range(0,Ncl):
        legend_names.append('class '+str(lbl))

    epochs = len(history['val_loss'])
    steps_per_valid = len(history['val_loss'][0])

    hist_loss_train = []
    hist_acc_train = []
    hist_loss_valid = []
    hist_acc_valid = []
    hist_f1 = []
    hist_recall = []
    hist_precision = []
    for e in range(epochs):
        hist_loss_train.append(np.mean(history['loss'         ][e][-steps_per_valid:]))
        hist_acc_train.append( np.mean(history['acc'          ][e][-steps_per_valid:]))

        hist_loss_valid.append(np.mean(history['val_loss'     ][e]))
        hist_acc_valid.append( np.mean(history['val_acc'      ][e]))

        # array_f1 = np.array(history['val_f1'       ][e]) # easier to achieve desired averaging (per class) with np array
        # array_re = np.array(history['val_recall'   ][e])
        # array_pr = np.array(history['val_precision'][e])
        hist_f1.append(        np.mean(np.array(history['val_f1'       ][e]), axis=0))
        hist_recall.append(    np.mean(np.array(history['val_recall'   ][e]), axis=0))
        hist_precision.append( np.mean(np.array(history['val_precision'][e]), axis=0))

    fig = plt.figure(figsize=(15,12))
    plt.subplot(321)
    plt.plot(hist_loss_train, label='train')
    plt.plot(hist_loss_valid, label='valid')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.grid()

    plt.subplot(323)
    plt.plot(hist_acc_train, label='train')
    plt.plot(hist_acc_valid, label='valid')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.grid()

    plt.subplot(322)
    plt.plot(hist_f1)
    plt.ylabel('F1-score')
    plt.xlabel('epochs')
    plt.legend(legend_names)
    plt.grid()

    plt.subplot(324)
    plt.plot(hist_precision)
    plt.ylabel('precision')
    plt.xlabel('epochs')
    plt.grid()

    plt.subplot(326)
    plt.plot(hist_recall)
    plt.ylabel('recall')
    plt.xlabel('epochs')
    plt.grid()

    fig.savefig(filename)

