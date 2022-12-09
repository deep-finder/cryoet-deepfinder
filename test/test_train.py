#import sys
#sys.path.append('../')

import os
import unittest
import shutil, tempfile
import numpy as np

from deepfinder.training import Train
from deepfinder.utils.common import write_array
import deepfinder.utils.objl as ol
from deepfinder.utils.dataloader import Dataloader

from test_utils import create_dummy_dset_for_train

# Create dummy dataset:
#mkdir('tmp/')
#write_array(np.random.rand())

# class TestTrainer(unittest.TestCase):
#
#     def test_launch(self):
#         prefix = '/net/serpico-fs2/emoebel/'
#
#         # Input parameters:
#         path_data = [prefix + 'cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/tomo0_data.mrc',
#                      prefix + 'cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/tomo1_data.mrc',]
#
#         path_target = [prefix + 'cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/tomo0_target.mrc',
#                        prefix + 'cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/tomo1_target.mrc',]
#
#         path_objl_train = prefix + 'cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/tomo0_objl.xml'
#         path_objl_valid = prefix + 'cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/tomo1_objl.xml'
#
#         Nclass = 16  #2
#         dim_in = 32  # patch size
#
#         # Initialize training task:
#         trainer = Train(Ncl=Nclass, dim_in=dim_in)
#         trainer.path_out = prefix + 'cryo/shrec2021/localization/test_deepfinder/test_dataloader/out/'  # output path
#         trainer.h5_dset_name = 'dataset'  # if training data is stored as h5, you can specify the h5 dataset
#         trainer.batch_size = 1
#         trainer.epochs = 1
#         trainer.steps_per_epoch = 1
#         trainer.Nvalid = 1  # steps per validation
#         trainer.flag_direct_read = False
#         trainer.flag_batch_bootstrap = True
#         trainer.Lrnd = 13  # random shifts when sampling patches (data augmentation)
#         trainer.class_weights = None  # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0
#
#         # Load object lists:
#         objl_train = ol.read_xml(path_objl_train)
#         objl_valid = ol.read_xml(path_objl_valid)
#
#         # Finally, launch the training procedure:
#         trainer.launch(path_data, path_target, objl_train, objl_valid)

# class TestTrainerWithDataloader(unittest.TestCase):
#
#     def test_launch(self):
#         prefix = '/net/serpico-fs2/emoebel/'
#
#         path_dset = '/net/serpico-fs2/emoebel/cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/'
#         path_data, path_target, objl_train, objl_valid = Dataloader()(path_dset)
#
#         # Input parameters:
#         Nclass = 16  #2
#         dim_in = 32  # patch size
#
#         # Initialize training task:
#         trainer = Train(Ncl=Nclass, dim_in=dim_in)
#         trainer.path_out = prefix + 'cryo/shrec2021/localization/test_deepfinder/test_dataloader/out/'  # output path
#         trainer.h5_dset_name = 'dataset'  # if training data is stored as h5, you can specify the h5 dataset
#         trainer.batch_size = 1
#         trainer.epochs = 1
#         trainer.steps_per_epoch = 1
#         trainer.Nvalid = 1  # steps per validation
#         trainer.flag_direct_read = False
#         trainer.flag_batch_bootstrap = True
#         trainer.Lrnd = 13  # random shifts when sampling patches (data augmentation)
#         trainer.class_weights = None  # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0
#
#         # Finally, launch the training procedure:
#         trainer.launch(path_data, path_target, objl_train, objl_valid)


class TestTrainerWithDataloaderWithDummyData(unittest.TestCase):

    def setUp(self):
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        # Create dummy dataset:
        create_dummy_dset_for_train(
            path=self.test_dir,
            n_tomos_train = 2,
            n_tomos_valid = 1,
            tomodim = (50, 100, 100),
            n_obj_classes = 2,
        )

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_launch(self):
        #prefix = '/net/serpico-fs2/emoebel/'
        #prefix = '/Users/emoebel/serpico-fs2/'

        #path_dset = prefix+'cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/'
        path_dset = self.test_dir
        path_data, path_target, objl_train, objl_valid = Dataloader()(path_dset)

        # Input parameters:
        Nclass = 2+1
        dim_in = 32  # patch size

        # Initialize training task:
        trainer = Train(Ncl=Nclass, dim_in=dim_in)
        trainer.path_out = self.test_dir  # output path
        trainer.h5_dset_name = 'dataset'  # if training data is stored as h5, you can specify the h5 dataset
        trainer.batch_size = 1
        trainer.epochs = 1
        trainer.steps_per_epoch = 1
        trainer.Nvalid = 1  # steps per validation
        trainer.flag_direct_read = False
        trainer.flag_batch_bootstrap = True
        trainer.Lrnd = 13  # random shifts when sampling patches (data augmentation)
        trainer.class_weights = None  # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0

        # Finally, launch the training procedure:
        trainer.launch(path_data, path_target, objl_train, objl_valid)

if __name__ == '__main__':
    unittest.main()
