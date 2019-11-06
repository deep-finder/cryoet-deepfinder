import sys
sys.path.append('../../') # add parent folder to path

import deepfind
import utils

path_data    = '../../data/tomo9.h5'
path_weights = '../training/params_model_FINAL.h5'

deepfind  = deepfind.deepfind(Ncl=13)

# Load data:
data = utils.load_h5array(path_data)
# Segment data:
scoremaps = deepfind.segment(data, path_weights)
# Get labelmap from scoremaps:
scoremaps = scoremaps[25:-25,25:-25,25:-25,:]
labelmap  = utils.scoremaps2labelmap(scoremaps)
# Bin labelmap for the clustering step (saves up computation time):
scoremapsB = utils.bin_scoremaps(scoremaps)
labelmapB  = utils.scoremaps2labelmap(scoremapsB)
# Save labelmaps:
utils.write_labelmap(labelmap , 'result/tomo9_labelmap.h5')
utils.write_labelmap(labelmapB, 'result/tomo9_bin1_labelmap.h5')

# Print out visualizations of the test tomogram and obtained segmentation:
utils.plot_volume_orthoslices(data[25:-25,25:-25,25:-25], 'result/tomo9_data.png')
utils.plot_volume_orthoslices(labelmap, 'result/tomo9_segmentation.png')