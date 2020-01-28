import numpy as np
import utils
import h5py

import matplotlib
matplotlib.use('agg') # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'
import matplotlib.pyplot as plt


# This functions loads the training set at specified paths.
# INPUTS:
#   path_data  : list of strings '/path/to/tomogram.ext'
#   path_target: list of strings '/path/to/target.ext'
#   dset_name  : can be usefull if files are stored as .h5
# OUTPUTS:
#   data_list  : list of 3D numpy arrays (tomograms)
#   target_list: list of 3D numpy arrays (annotated tomograms)
def load_dataset(path_data, path_target, dset_name='dataset'):
    data_list   = []
    target_list = []
    for idx in range(0,len(path_data)):
        data_list.append(  utils.read_array(path_data[idx]  , dset_name))
        target_list.append(utils.read_array(path_target[idx], dset_name))
    return data_list, target_list


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
        bs_idx.append( np.random.choice(np.squeeze(np.asarray(np.nonzero(label_list==l))), Nbs) )
    bs_idx = np.concatenate(bs_idx)
    return bs_idx
    
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
    if (x>tomodim[0]-p_in): x = tomodim[0]-p_in
    if (y>tomodim[1]-p_in): y = tomodim[1]-p_in
    if (z>tomodim[2]-p_in): z = tomodim[2]-p_in

    #else: # sample random position in tomogram
    #    x = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    #    y = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    #    z = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    
    return x,y,z
    
def save_history(history, filename):
    h5file = h5py.File(filename, 'w')

    # train and val loss & accuracy:
    dset    = h5file.create_dataset('acc', (len(history['acc']),))
    dset[:] = history['acc']
    dset    = h5file.create_dataset('loss', (len(history['loss']),))
    dset[:] = history['loss']
    dset    = h5file.create_dataset('val_acc', (len(history['val_acc']),))
    dset[:] = history['val_acc']
    dset    = h5file.create_dataset('val_loss', (len(history['val_loss']),))
    dset[:] = history['val_loss']

    # val precision, recall, F1:
    dset    = h5file.create_dataset('val_f1', np.shape(history['val_f1']))
    dset[:] = history['val_f1']
    dset    = h5file.create_dataset('val_precision', np.shape(history['val_precision']))
    dset[:] = history['val_precision']
    dset    = h5file.create_dataset('val_recall', np.shape(history['val_recall']))
    dset[:] = history['val_recall']

    h5file.close()
    return
    
def plot_history(history, filename):
    Ncl = len(history['val_f1'][0])
    legend_names = []
    for lbl in range(0,Ncl):
        legend_names.append('class '+str(lbl))

    fig = plt.figure(figsize=(15,12))
    plt.subplot(321)
    plt.plot(history['loss']    , label='train')
    plt.plot(history['val_loss'], label='valid')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.grid()

    plt.subplot(323)
    plt.plot(history['acc']    , label='train')
    plt.plot(history['val_acc'], label='valid')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.grid()

    plt.subplot(322)
    plt.plot(history['val_f1'])
    plt.ylabel('F1-score')
    plt.xlabel('epochs')
    plt.legend(legend_names)
    plt.grid()

    plt.subplot(324)
    plt.plot(history['val_precision'])
    plt.ylabel('precision')
    plt.xlabel('epochs')
    plt.grid()

    plt.subplot(326)
    plt.plot(history['val_recall'])
    plt.ylabel('recall')
    plt.xlabel('epochs')
    plt.grid()

    fig.savefig(filename)


# Following observer classes are needed to send prints to GUI if needed:
class observer_print:
    def display(message):
        print(message)

class observer_gui:
    def __init__(self, pyqt_signal):
        self.sig = pyqt_signal
    def display(self, message):
        self.sig.emit(message)
