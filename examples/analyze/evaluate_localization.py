import sys
sys.path.append('../../') # add parent folder to path

import numpy as np
import h5py
import utils

import matplotlib
matplotlib.use('agg') # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'
import matplotlib.pyplot as plt

Ncl_obj = 9  # number of object classes (dont count background as class here)
Ntomo   = 10

tol_pos_err = 5

path_objl_gt = '../../data/'
path_objl_df = 'result/'

recall    = np.zeros((Ncl_obj,))
precision = np.zeros((Ncl_obj,))
f1        = np.zeros((Ncl_obj,))
for l in range(0,Ncl_obj):
    label = l+1
    print('Evaluating class '+str(label)+' ...')
    Ngt_master = 0
    Ndf_master = 0
    Ntp_master = 0
    for T in range(0,Ntomo):
        objl_gt = utils.read_objlist(path_objl_gt+'tomo'+str(T+1)+'_objlist.xml')
        objl_df = utils.read_objlist(path_objl_df+'tomo'+str(T+1)+'_objlist.xml')
        
        objl_gt = utils.objlist_get_class(objl_gt, label)
        objl_df = utils.objlist_get_class(objl_df, label)
        
        objl_df = utils.objlist_above_thr(objl_df, 10) # remove clusters smaller than 10 voxels (considered as false positives)
        
        Ngt = len(objl_gt)
        Ndf = len(objl_df)
        Ntp = utils.get_Ntp_from_objlist(objl_gt, objl_df, tol_pos_err)
        
        Ngt_master = Ngt_master + Ngt
        Ndf_master = Ndf_master + Ndf
        Ntp_master = Ntp_master + Ntp
    recall[l]    = float(Ntp)/float(Ngt)
    precision[l] = float(Ntp)/float(Ndf)
    f1[l]        = 2*(recall[l]*precision[l])/(recall[l]+precision[l])
    
# Plot:
index = range(1,Ncl_obj+1)
fig = plt.figure(figsize=(10,7))
plt.subplot(311)
plt.bar(index, f1, color='r')
plt.xlabel('class label')
plt.ylabel('F1-score')
plt.grid()

plt.subplot(312)
plt.bar(index, recall, color='b')
plt.xlabel('class label')
plt.ylabel('recall')
plt.grid()

plt.subplot(313)
plt.bar(index, precision, color='g')
plt.xlabel('class label')
plt.ylabel('precision')
plt.grid()

fig.savefig('scores.png')