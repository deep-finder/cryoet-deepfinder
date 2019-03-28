# This script prints out visualizations of the tomogram, the segmentation ground truth, and the segmentation obtained by deep finder.

import sys
sys.path.append('../../') # add parent folder to path

import numpy as np
import h5py
import utils

data   = utils.load_h5array('../../data/tomo1_data.h5')
target = utils.load_h5array('../../data/tomo1_target.h5')
pred   = utils.load_h5array('result/tomo1_labelmap.h5')

utils.plot_volume_orthoslices(data  , 'tomo1_data.png')
utils.plot_volume_orthoslices(target, 'tomo1_groundtruth.png')
utils.plot_volume_orthoslices(pred  , 'tomo1_segmentation.png')