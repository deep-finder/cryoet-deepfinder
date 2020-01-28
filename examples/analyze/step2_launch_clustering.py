import sys
sys.path.append('../../') # add parent folder to path

import deepfind as df
import utils
import utils_objl as ol

# Input parameters:
path_labelmap = 'out/tomo9_bin1_labelmap.mrc'
cluster_radius = 5         # should correspond to average radius of target objects (in voxels)
cluster_size_threshold = 1 # found objects smaller than this threshold are immediately discarded

# Output parameter:
path_output = 'out/'


# Load data:
labelmapB = utils.read_array(path_labelmap)

# Initialize clustering task:
clust = df.Cluster(clustRadius=5)
clust.sizeThr = cluster_size_threshold

# Launch clustering (result stored in objlist): can take some time (37min on i7 cpu)
objlist = clust.launch(labelmapB)


# Post-process the object list for evaluation:

# The coordinates have been obtained from a binned (subsampled) volume, therefore coordinates have to be re-scaled in
# order to compare to ground truth:
objlist = ol.scale_coord(objlist, 2)

# Then, we filter out particles that are too small, considered as false positives. As macromolecules have different
# size, each class has its own size threshold. The thresholds have been determined on the validation set.
lbl_list = [ 1,   2,  3,   4,  5,   6,   7,  8,  9, 10,  11, 12 ]
thr_list = [50, 100, 20, 100, 50, 100, 100, 50, 50, 20, 300, 300]

objlist_thr = ol.above_thr_per_class(objlist, lbl_list, thr_list)

# Save object lists:
ol.write_xml(objlist    , path_output+'tomo9_objlist_raw.xml')
ol.write_xml(objlist_thr, path_output+'tomo9_objlist_thr.xml')