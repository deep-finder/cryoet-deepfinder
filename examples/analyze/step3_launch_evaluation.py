# This script computes recall, precision and f1-score for each object class, and prints out the result in log files.
# The evaluation is based on a script used for the challenge "SHREC 2019: Classification in cryo-electron tomograms"

# This script needs python3 and additional packages (see evaluate.py), as it was coded by SHREC'19 organizers. The 
# scores have been published in Gubins & al., "SHREC'19 track: classification in cryo-electron tomograms", 2019

import sys
sys.path.append('../../') # add parent folder to path

import os
import deepfinder.utils.objl as ol

# First, we load the object list produced by DeepFinder:
objl = ol.read_xml('result/tomo9_objlist_thresholded.xml')

# Then, we convert the predicted object list into a text file, as needed by the SHREC'19 evaluation script:
class_name = {0: "0", 1: "1bxn", 2: "1qvr", 3: "1s3x", 4: "1u6g", 5: "2cg9", 6: "3cf3",
                       7: "3d2f", 8: "3gl1", 9: "3h84", 10: "3qm1", 11: "4b4t", 12: "4d8q"}
file = open('result/particle_locations_tomo9.txt', 'w')
for p in range(0,len(objl)):
    x   = int( objl[p]['x'] )
    y   = int( objl[p]['y'] )
    z   = int( objl[p]['z'] )
    lbl = int( objl[p]['label'] )
    file.write(class_name[lbl]+' '+str(x)+' '+str(y)+' '+str(z)+'\n')
file.close()

# Finally, we launch the SHREC'19 evaluation script:
os.system('python3 evaluate.py --gtcoordinates -f result/particle_locations_tomo9.txt -o result/ -tf ../../data/')
