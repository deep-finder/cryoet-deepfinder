import sys
sys.path.append('../../') # add parent folder to path

from deepfinder.training import Train
import deepfinder.utils.objl as ol
from deepfinder.utils.dataloader import Dataloader

# This script will not work because this repository does not include the training set. However it shows how training
# is realized.

# Load dataset:
path_dset = '/path/to/dataset/folder/'
path_data, path_target, objl_train, objl_valid = Dataloader()(path_dset)

# Input parameters:
Nclass = 13
dim_in = 56  # patch size

# Initialize training task:
trainer = Train(Ncl=Nclass, dim_in=dim_in)
trainer.path_out         = 'out/' # output path
trainer.h5_dset_name     = 'dataset' # if training data is stored as h5, you can specify the h5 dataset
trainer.batch_size       = 25
trainer.epochs           = 100
trainer.steps_per_epoch  = 100
trainer.Nvalid           = 10 # steps per validation
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True
trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)
trainer.class_weights = None # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0

# Use following line if you want to resume a previous training session:
#trainer.net.load_weights('out/round1/net_weights_FINAL.h5')

# Finally, launch the training procedure:
trainer.launch(path_data, path_target, objl_train, objl_valid)
