import sys
sys.path.append('../../') # add parent folder to path

import deepfind
import utils

path_data    = '../../data/'
path_weights = '../training/params_model_FINAL.h5'

Ntomo = 10
deepfind  = deepfind.deepfind(Ncl=10)

# Segment all 10 test tomograms:
for T in range(1,Ntomo+1):
    print('Segmenting tomogram '+str(T)+' ...')
    # Load data:
    data = utils.load_h5array(path_data+'tomo'+str(T)+'_data.h5')
    # Segment data:
    scoremaps = deepfind.segment_single_block(data, path_weights)
    # Get labelmap from scoremaps:
    labelmap  = utils.scoremaps2labelmap(scoremaps)
    # Save labelmap:
    utils.write_labelmap(labelmap, 'result/tomo'+str(T)+'_labelmap.h5')

# Note: here I use the function 'segment_single_block'. However, this function restricted to input volumes of cubic shape, which is the case here. For testing on your own data I recomment using the function 'segment'.