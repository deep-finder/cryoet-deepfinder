import sys
sys.path.append('../../') # add parent folder to path

import numpy as np
import deepfind as df
import utils
import utils_objl as ol

path_output = './'

# First load object list with annotated positions. There are two strategies to generate targets: shapes and spheres
# If you choose 'shapes', the object list has to contain the orientation (i.e. Euler angles) of the objects.
objl = ol.read_xml('object_list_train.xml') # contains objects of whole dataset

# Let's generate the target for tomogram 0:
objl = ol.get_tomo(objl, 0) # now contains objects of tomo0 only

# Set initial volume: can be used to add segmented structures to target (e.g. membranes). If not, simply initialize with
# empty volume (zero values)
tomodim = (200, 512, 512)
initial_vol = np.zeros(tomodim)

# Initialize target generation task:
tbuild = df.TargetBuilder()
# Launch target generation:

# # For 'shapes' strategy:
# ref_list = ['/path/mask_class1.mrc',
#             '/path/mask_class2.mrc',
#             '/path/mask_class3.mrc']
# target = tbuild.generate_with_shapes(objl, initial_vol, ref_list)
# utils.plot_volume_orthoslices(target, path_output+'orthoslices_target_shapes.png')

# For 'spheres' strategy:
radius_list = [6, 6, 3, 6, 6, 7, 6, 4, 4, 3, 10, 8]
target = tbuild.generate_with_spheres(objl, initial_vol, radius_list)
utils.plot_volume_orthoslices(target, path_output+'orthoslices_target_spheres.png')

# Save target:
utils.write_array(target, path_output+'target_tomo0.mrc')