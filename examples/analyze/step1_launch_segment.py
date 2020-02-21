import sys
sys.path.append('../../') # add parent folder to path

from deepfinder.inference import Segment
import deepfinder.utils.common as cm
import deepfinder.utils.smap as sm

# Input parameters:
path_tomo    = 'in/tomo9.mrc' # tomogram to be segmented
path_weights = 'in/net_weights_FINAL.h5' # weights for neural network (obtained from training)
Nclass       = 13  # including background class
patch_size   = 160 # must be multiple of 4

# Output parameter:
path_output = 'out/'


# Load data:
tomo = cm.read_array(path_tomo)

# Initialize segmentation task:
seg  = Segment(Ncl=Nclass, path_weights=path_weights, patch_size=patch_size)

# Segment tomogram:
scoremaps = seg.launch(tomo)

# Get labelmap from scoremaps:
labelmap  = sm.to_labelmap(scoremaps)

# Bin labelmap for the clustering step (saves up computation time):
scoremapsB = sm.bin(scoremaps)
labelmapB  = sm.to_labelmap(scoremapsB)

# Save labelmaps:
cm.write_array(labelmap , path_output+'tomo9_labelmap.mrc')
cm.write_array(labelmapB, path_output+'tomo9_bin1_labelmap.mrc')

# Print out visualizations of the test tomogram and obtained segmentation:
cm.plot_volume_orthoslices(tomo    , path_output+'orthoslices_tomo9.png')
cm.plot_volume_orthoslices(labelmap, path_output+'orthoslices_tomo9_segmentation.png')
