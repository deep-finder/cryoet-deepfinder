import sys
sys.path.append('../../') # add parent folder to path

import deepfind as df
import utils_objl as ol

# Input parameters:
# path_data = ['/path/tomo1.mrc',
#              '/path/tomo2.mrc',
#              '/path/tomo3.mrc']
#
# path_target = ['/path/target1.mrc',
#                '/path/target2.mrc',
#                '/path/target3.mrc']
prefix = '/net/serpico-fs2/emoebel/shrec2019/data/for_cnn/'
path_data   = []
path_target = []
for idx in range(0,9):
    path_data.append(prefix+'tomo'+str(idx)+'/data.h5')
    path_target.append(prefix+'tomo'+str(idx)+'/target_thr1.h5')

path_objl_train = 'in/object_list_train.xml'
path_objl_valid = 'in/object_list_valid.xml'

Nclass = 13

# Initialize training task:
trainer = df.Train(Ncl=Nclass)
trainer.path_out         = 'out/' # output path
trainer.dim_in           = 56 # patch size
trainer.batch_size       = 25
trainer.epochs           = 2
trainer.steps_per_epoch  = 2
trainer.Nvalid           = 2 # steps per validation
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True
trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)

# Load object lists:
objl_train = ol.read_xml(path_objl_train)
objl_valid = ol.read_xml(path_objl_valid)

# Finally, launch the training procedure:
trainer.launch(path_data, path_target, objl_train, objl_valid)
