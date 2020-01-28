import sys
sys.path.append('../../') # add parent folder to path

import deepfind as df
import utils
import utils_smap as sm

# Input parameters:
path_tomo    = 'in/tomo9.mrc' # tomogram to be segmented
path_weights = 'in/net_weights_FINAL.h5' # weights for neural network (obtained from training)
Nclass       = 13  # including background class
patch_size   = 160 # must be multiple of 4

# Output parameter:
path_output = 'out/'


# Load data:
tomo = utils.read_array(path_tomo)

# Initialize segmentation task:
seg  = df.Segment(Ncl=Nclass, path_weights=path_weights)

# Segment tomogram:
scoremaps = seg.launch(tomo)

# Get labelmap from scoremaps:
labelmap  = sm.to_labelmap(scoremaps)

# Bin labelmap for the clustering step (saves up computation time):
scoremapsB = sm.bin(scoremaps)
labelmapB  = sm.to_labelmap(scoremapsB)

# Save labelmaps:
utils.write_array(labelmap , path_output+'tomo9_labelmap.mrc')
utils.write_array(labelmapB, path_output+'tomo9_bin1_labelmap.mrc')
utils.write_array(labelmapB, path_output+'tomo9_bin1_labelmap.mrc')

# Print out visualizations of the test tomogram and obtained segmentation:
utils.plot_volume_orthoslices(tomo    , path_output+'orthoslices_tomo9.png')
utils.plot_volume_orthoslices(labelmap, path_output+'orthoslices_tomo9_segmentation.png')



# # Load tomogram:
# data = utils.read_array(path_data)
#
# deepfind  = deepfind.deepfind(Ncl=13)
#
# # Segment data:
# scoremaps = deepfind.segment(data, path_weights)
# # Get labelmap from scoremaps:
# scoremaps = scoremaps[25:-25,25:-25,25:-25,:]
# labelmap  = utils.scoremaps2labelmap(scoremaps)
# # Bin labelmap for the clustering step (saves up computation time):
# scoremapsB = utils.bin_scoremaps(scoremaps)
# labelmapB  = utils.scoremaps2labelmap(scoremapsB)
# # Save labelmaps:
# utils.write_labelmap(labelmap , 'result/tomo9_labelmap.h5')
# utils.write_labelmap(labelmapB, 'result/tomo9_bin1_labelmap.h5')
#
# # Print out visualizations of the test tomogram and obtained segmentation:
# utils.plot_volume_orthoslices(data[25:-25,25:-25,25:-25], 'result/tomo9_data.png')
# utils.plot_volume_orthoslices(labelmap, 'result/tomo9_segmentation.png')