import sys
sys.path.append('../../') # add parent folder to path

import deepfind
import utils

Ntomo = 10
deepfind  = deepfind.deepfind(Ncl=10)

for T in range(1,Ntomo+1):
    print('Clustering labelmap '+str(T)+' ...')
    # Load labelmap:
    labelmap = utils.load_h5array('result/tomo'+str(T)+'_labelmap.h5')
    # Launch clustering (result stored in objlist):
    objlist = deepfind.cluster(labelmap, sizeThr=1, clustRadius=8)
    # Save objlist:
    utils.write_objlist(objlist, 'result/tomo'+str(T)+'_objlist.xml') 

