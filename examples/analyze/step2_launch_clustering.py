import sys
sys.path.append('../../') # add parent folder to path

import deepfind
import utils

from lxml import etree
from copy import deepcopy


deepfind  = deepfind.deepfind(Ncl=13)

# Load data:
labelmapB = utils.load_h5array('result/tomo9_bin1_labelmap.h5')

# Launch clustering (result stored in objlist):
objlist = deepfind.cluster(labelmapB, sizeThr=1, clustRadius=5)


# Post-process the objlist for evaluation:
# The coordinates have been obtained from a binned (subsampled) volume, therefore coordinates have to be re-scaled in order to compare to ground truth:
objlist = utils.objlist_scale_coord(objlist, 2)

# Then, we filter out particles that are too small, considered as false positives. As macromolecules have different size, each class has its own size threshold. The thresholds have been determined on the validation set.
lbl_list = [ 1,   2,  3,   4,  5,   6,   7,  8,  9, 10,  11, 12 ]
thr_list = [50, 100, 20, 100, 50, 100, 100, 50, 50, 20, 300, 300]

objlist_thr = etree.Element('objlist')
for lbl in lbl_list:
    objlist_class = utils.objlist_get_class(objlist, lbl)
    objlist_class = utils.objlist_above_thr(objlist_class, thr_list[lbl-1])
    for p in range(0,len(objlist_class)):
        objlist_thr.append( deepcopy(objlist_class[p]) )

# Save objlist:
utils.write_objlist(objlist    , 'tomo9_objlist_raw.xml') 
utils.write_objlist(objlist_thr, 'result/tomo9_objlist_thresholded.xml') 