import sys
sys.path.append('../../') # add parent folder to path

import numpy as np
from deepfinder.training import TargetBuilder
import deepfinder.utils.common as cm
import deepfinder.utils.objl as ol

path_output = 'out/'

# First load object list with annotated positions. There are two strategies to generate targets: shapes and spheres
# If you choose 'shapes', the object list has to contain the orientation (i.e. Euler angles) of the objects.
objl = ol.read_xml('in/object_list_tomo0.xml')

# Set initial volume: can be used to add segmented structures to target (e.g. membranes). If not, simply initialize with
# empty volume (zero values)
tomodim = (200, 512, 512)
initial_vol = np.zeros(tomodim)

# Initialize target generation task:
tbuild = TargetBuilder()

# Launch target generation:
# For 'spheres' strategy:
radius_list = [6, 6, 3, 6, 6, 7, 6, 4, 4, 3, 10, 8]
target = tbuild.generate_with_spheres(objl, initial_vol, radius_list)
cm.plot_volume_orthoslices(target, path_output+'orthoslices_target_spheres.png')

# # For 'shapes' strategy:
# pdb_name = ['1bxn', '1qvr', '1s3x', '1u6g', '2cg9', '3cf3', '3d2f', '3gl1', '3h84', '3qm1', '4b4t', '4d8q']
# ref_list = []
# for pdb in pdb_name:
#     mask = cm.read_array('in/masks/'+pdb+'.mrc')
#     ref_list.append(mask)
#
# target = tbuild.generate_with_shapes(objl, initial_vol, ref_list)
# cm.plot_volume_orthoslices(target, path_output+'orthoslices_target_shapes.png')

# Save target:
cm.write_array(target, path_output+'target_tomo0.mrc')