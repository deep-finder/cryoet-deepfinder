import sys
sys.path.append('../')

import os
import unittest
import numpy as np

from cryoet-deepfinder.training import Train
from cryoet-deepfinder.utils.common import write_array

# Create dummy dataset:
mkdir('tmp/')
write_array(np.random.rand())

class TestTrainer(unittest.TestCase):

    def test_launch(self):
        # Input parameters:
        path_data = ['/path/tomo1.mrc',
                     '/path/tomo2.mrc',
                     '/path/tomo3.mrc']

        path_target = ['/path/target1.mrc',
                       '/path/target2.mrc',
                       '/path/target3.mrc']

        path_objl_train = 'in/object_list_train.xml'
        path_objl_valid = 'in/object_list_valid.xml'

        Nclass = 2
        dim_in = 32  # patch size

        # Initialize training task:
        trainer = Train(Ncl=Nclass, dim_in=dim_in)
        trainer.path_out = 'out/'  # output path
        trainer.h5_dset_name = 'dataset'  # if training data is stored as h5, you can specify the h5 dataset
        trainer.batch_size = 1
        trainer.epochs = 1
        trainer.steps_per_epoch = 1
        trainer.Nvalid = 1  # steps per validation
        trainer.flag_direct_read = False
        trainer.flag_batch_bootstrap = True
        trainer.Lrnd = 13  # random shifts when sampling patches (data augmentation)
        trainer.class_weights = None  # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0

        # Load object lists:
        objl_train = ol.read_xml(path_objl_train)
        objl_valid = ol.read_xml(path_objl_valid)

        # Finally, launch the training procedure:
        trainer.launch(path_data, path_target, objl_train, objl_valid)

