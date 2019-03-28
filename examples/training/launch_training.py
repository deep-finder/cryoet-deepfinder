import sys
sys.path.append('../../../') # add parent folder to path

import deepfind
import utils

# First, initialize the Deep Finder framework:
deepfind = deepfind.deepfind(Ncl=10)

# Note: all training parameters are set to default values. If you wish to modify a parameter, for example the number of epochs, you can do so as follows : deepfind.epochs = 300

# Then, prepare the arguments necessary for the training procedure (data paths and object lists):
prefix = '/path/to/dataset/'
path_data   = []
path_target = []
for idx in range(0,250):
    path_data.append(prefix+'tomo'+str(idx+1)+'_data.h5')
    path_target.append(prefix+'tomo'+str(idx+1)+'_target.h5')

objlist_train = utils.read_objlist('object_list_train.xml')
objlist_valid = utils.read_objlist('object_list_valid.xml')

# Finally, launch the training procedure:
deepfind.train(path_data, path_target, objlist_train, objlist_valid)

# Note on data format: the procedure expects the tomogram and target volumes to be stored in .h5 files (may be extended in future versions). Inside the h5 file, the array should be stored in a dataset named 'dataset'.
