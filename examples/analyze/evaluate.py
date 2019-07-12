import mrcfile as mrc
import numpy as np
import warnings
from pathlib import Path
from pycm import ConfusionMatrix
from collections import OrderedDict
from scipy.spatial import distance
from contextlib import redirect_stdout
import argparse
import json
warnings.simplefilter('ignore') # for mrcfile

### How to use evaluation script
# First, install dependencies: pycm, scipy, mrcfile ... (all should be available through PIP)
# Then, run the script with correct arguments with Python 3
#
### Customization that was used for evaluation:
# submission 2, 6 and 7 provided results in GT coordinates (Z slice 0-200) instead of tomo coordinates (Z slice 0-512)
# submission 4 asked to remove 10 from each dimension
#
### Usual example
# File with predicted particles: C:\shrec\submission.txt
# Output folder: C:\shrec\
# Path to 9th folder of the dataset: C:\shrec\dataset\9
# Script to run: python3 evaluate.py -f C:\shrec\submission.txt -o C:\shrec\ -tf C:\shrec\dataset\9
#
### Example with padding of 10 and in GT coordinates
# File with predicted particles: C:\shrec\submission.txt
# Output folder: C:\shrec\
# Path to 9th folder of the dataset: C:\shrec\dataset\9
# Script to run: python3 evaluate.py --gtcoordinates -p 10 -f C:\shrec\submission.txt -o C:\shrec\ -tf C:\shrec\dataset\9
#

############################################################################# Argparser
parser = argparse.ArgumentParser(description='SHREC Cryo-ET 2019 evaluation script')

parser.add_argument('-f', '--file', help='Path to predicted particles file',
                    required=True, type=str)
parser.add_argument('-o', '--output', help='Path to output folder, where evaluation will be saved',
                    required=True, type=str)
parser.add_argument('-tf', '--tomofolder', help='Path to folder with test tomogram and hitbox volume',
                    required=True, type=str)
### Customization of evaluation
parser.add_argument('--gtcoordinates', default=True, action='store_false',
                    help='If flag is used, predicted particles are considered to be in GT coordinates (Z slices 0-200),'
                         'instead of tomo coordinates (Z slices 0-512)')
parser.add_argument('-p', '--padding', help='If provided, essentially substracts the value from all'
                                            'coordinates of the provided particles', default=0, type=int)
args = parser.parse_args()

############################################################################# paths
results_file = Path(args.file)
tomo_folder = Path(args.tomofolder)
evaluation_folder = Path(args.output)
gt_particles_path = tomo_folder / 'particle_locations_model_9.txt'
hitbox_path = tomo_folder / 'hitbox_9.mrc'

############################################################################# load global dicts for class conversions
num2pdb = OrderedDict({0: "0", 1: "1bxn", 2: "1qvr", 3: "1s3x", 4: "1u6g", 5: "2cg9", 6: "3cf3",
                       7: "3d2f", 8: "3gl1", 9: "3h84", 10: "3qm1", 11: "4b4t", 12: "4d8q"})
pdb2num = OrderedDict({"0": 0, "1bxn": 1, "1qvr": 2, "1s3x": 3, "1u6g": 4, "2cg9": 5, "3cf3": 6,
                       "3d2f": 7, "3gl1": 8, "3h84": 9, "3qm1": 10, "4b4t": 11, "4d8q": 12})

############################################################################# load GT data
with mrc.open(str(hitbox_path), permissive=True) as f:
    hitbox = f.data
gt_shape = hitbox.shape
gt_particles = []
gt_z_offset = 156
with open(gt_particles_path, 'rU') as f:
    for line in f:
        pdb_id, Z, Y, X, rot_1, rot_2, rot_3 = line.rstrip('\n').split()
        gt_particles.append((pdb_id, int(Z), int(Y), int(X)))
gt_particle_classes = np.asarray([pdb2num[p[0]] for p in gt_particles])

############################################################################# loading submission / predicted results
predicted_particles = []
with open(str(results_file), 'rU') as f:
    for line in f:
        pdb, Z, Y, X, *_ = line.rstrip('\n').split()
        predicted_particles.append((pdb, int(round(float(Z))), int(round(float(Y))), int(round(float(X)))))

############################################################################# stats placeholders
# for stats - will be populated during walk through predicted particles
predicted_particle_hit_count = np.zeros((len(gt_particles), ), dtype=int)
predicted_particle_hit_count_class = {k: 0 for k, v in pdb2num.items()}
predicted_particle_classes = np.zeros_like(gt_particle_classes)
total_hits = 0
total_distance = 0
clipped_particles = 0

############################################################################# convert coordinates
gt_z_offset = args.gtcoordinates * gt_z_offset
pad = args.padding

predicted_particles = [(p[0], p[1] - gt_z_offset - pad, p[2] - pad, p[3] - pad) for p in predicted_particles]

############################################################################# go through particles
for i, p in enumerate(predicted_particles):
    pdb, Z, Y, X = p

    # clamp coordinates to fit to ground truth (some submission had indices of 512+)
    z = max(min(Z, gt_shape[0] - 1), 0)
    y = max(min(Y, gt_shape[1] - 1), 0)
    x = max(min(X, gt_shape[2] - 1), 0)

    if Z != z or Y != y or X != x:
        clipped_particles += 1

    # find id of a particle at that location
    particle = int(hitbox[z][y][x])

    # there is no particle at that location: miss
    if particle == 0:
        predicted_particle_hit_count_class['0'] += 1
        continue

    particle -= 1 # particles in evaluation arrays are 0-indexed, but 1-indexed in hitbox (0 = nothing there)
    predicted_particle_hit_count[particle] += 1 # this particle was detected
    predicted_particle_hit_count_class[pdb] += 1 # particle of this class was detected

    # increase total particle hit counter
    total_hits += 1

    # find ground truth center
    real_particle = gt_particles[particle] # ground truth particle
    true_center = (real_particle[1], real_particle[2], real_particle[3])

    # compute euclidean distance from predicted center to real center
    total_distance += np.abs(distance.euclidean((z, y, x), true_center))

    # use only the first classification prediction for that particle
    if predicted_particle_classes[particle] == 0:
        predicted_particle_classes[particle] = pdb2num[pdb]

############################################################################# statistics
unique_particles_found = sum([int(p >= 1) for p in predicted_particle_hit_count])
unique_particles_not_found = sum([int(p == 0) for p in predicted_particle_hit_count])
multiple_hits = sum([int(p > 1) for p in predicted_particle_hit_count])

total_recall = unique_particles_found / len(gt_particles)
total_precision = unique_particles_found / len(predicted_particles)
total_f1 = 1 / ((1/total_recall + 1/total_precision) / 2)
total_missrate = unique_particles_not_found / len(gt_particles)
avg_distance = total_distance / total_hits

cm = ConfusionMatrix(actual_vector=gt_particle_classes, predict_vector=predicted_particle_classes)
# relabel confusion matrix
class_labels = num2pdb.copy()
# check whether all classes are represented and if not, remove them from label dict
for k in num2pdb:
    if k not in cm.classes:
        class_labels.pop(k)
cm.relabel(class_labels)


############################################################################# print to log
with open(str(evaluation_folder / 'evaluation_log.txt'), 'w') as f:
    with redirect_stdout(f):

        print('############################## LOCALIZATION EVALUATION')
        print(f'Found {len(predicted_particles)} results')
        print(f'TP: {unique_particles_found} unique particles localized out of total {len(gt_particles)} particles')
        print(f'FP: {predicted_particle_hit_count_class["0"]} reported particles are false positives')
        print(f'FN: {unique_particles_not_found} unique particles not found')
        if multiple_hits:
            print(f'Note: there were {multiple_hits} unique particles that had more than one result')
        if clipped_particles:
            print(f'Note: there were {clipped_particles} results that were outside of tomo bounds ({gt_shape})')
        print(f'Average euclidean distance from predicted center to ground truth center: {avg_distance:.5f}')
        print(f'Total recall: {total_recall:.5f}')
        print(f'Total precision: {total_precision:.5f}')
        print(f'Total miss rate: {total_missrate:.5f}')
        print(f'Total f1-score: {total_f1:.5f}')
        print('\n############################## CLASSIFICATION EVALUATION')
        print(cm)

############################################################################# print to stdout
print('############################## LOCALIZATION EVALUATION')
print(f'Found {len(predicted_particles)} results')
print(f'TP: {unique_particles_found} unique particles localized out of total {len(gt_particles)} particles')
print(f'FP: {predicted_particle_hit_count_class["0"]} reported particles are false positives')
print(f'FN: {unique_particles_not_found} unique particles not found')
if multiple_hits:
    print(f'Note: there were {multiple_hits} unique particles that had more than one result')
if clipped_particles:
    print(f'Note: there were {clipped_particles} results that were outside of tomo bounds ({gt_shape})')
print(f'Average euclidean distance from predicted center to ground truth center: {avg_distance:.5f}')
print(f'Total recall: {total_recall:.5f}')
print(f'Total precision: {total_precision:.5f}')
print(f'Total miss rate: {total_missrate:.5f}')
print(f'Total f1-score: {total_f1:.5f}')
print('\n############################## CLASSIFICATION EVALUATION')
print(cm)
print(f'\n############################## DONE.\nThe same report is saved in {str(evaluation_folder)}')

############################################################################# save confusion matrix reports
cm.save_csv(str(evaluation_folder / 'classification_log'))
cm.save_html(str(evaluation_folder / 'classification_log')) # for easier preview, save html too
