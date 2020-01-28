import h5py
import numpy as np
import time

from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import MeanShift

import models
import losses

import utils
import core_utils
import utils_objl as ol

# TODO: add path_output as class argument
class DeepFinder:
    def __init__(self):
        self.obs_list = [core_utils.observer_print]

    # Useful for sending prints to GUI
    def set_observer(self, obs):
        self.obs_list.append(obs)

    # "Master print" calls all observers for prints
    def display(self, message):
        for obs in self.obs_list: obs.display(message)


# import time
# class dummy(df):
#     def __init__(self):
#         df.__init__(self)
#     def launch(self):
#         for idx in range(5):
#             self.display('Iteration '+str(idx)+' ...')
#             time.sleep(1)

class TargetBuilder(DeepFinder):
    def __init__(self):
        DeepFinder.__init__(self)

    # Generates segmentation targets from object list. Here macromolecules are annotated with their shape.
    # INPUTS
    #   objl: list of dictionaries. Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #   ref_list: list of binary 3D arrays (expected to be cubic). These reference arrays contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
    #             The references order in list should correspond to the class label
    #             For ex: 1st element of list -> reference of class 1
    #                     2nd element of list -> reference of class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_shapes(self, objl, target_array, ref_list):
        N = len(objl)
        dim = target_array.shape
        for p in range(len(objl)):
            self.display('Annotating object ' + str(p + 1) + ' / ' + str(N) + ' ...')
            lbl = int(objl[p]['label'])
            x = int(objl[p]['x'])
            y = int(objl[p]['y'])
            z = int(objl[p]['z'])
            phi = objl[p]['phi']
            psi = objl[p]['psi']
            the = objl[p]['the']

            ref = ref_list[lbl - 1]
            centeroffset = np.int(np.floor(ref.shape[0] / 2))

            # Rotate ref:
            if phi!=None and psi!=None and the!=None:
                ref = utils.rotate_array(ref, (phi, psi, the))
                ref = np.int8(np.round(ref))

            # Get the coordinates of object voxels in target_array
            obj_voxels = np.nonzero(ref == 1)
            x_vox = obj_voxels[0] + x - centeroffset +1
            y_vox = obj_voxels[1] + y - centeroffset +1
            z_vox = obj_voxels[2] + z - centeroffset +1

            for idx in range(x_vox.size):
                xx = x_vox[idx]
                yy = y_vox[idx]
                zz = z_vox[idx]
                if xx >= 0 and xx < dim[0] and yy >= 0 and yy < dim[1] and zz >= 0 and zz < dim[2]:  # if in tomo bounds
                    target_array[xx, yy, zz] = lbl
        return np.int8(target_array)

    # Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
    # This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
    # On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'
    # INPUTS
    #   objl: list of dictionaries. Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #   radius_list: list of sphere radii (in voxels).
    #             The radii order in list should correspond to the class label
    #             For ex: 1st element of list -> sphere radius for class 1
    #                     2nd element of list -> sphere radius for class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_spheres(self, objl, target_array, radius_list):
        Rmax = max(radius_list)
        dim = [2*Rmax, 2*Rmax, 2*Rmax]
        ref_list = []
        for idx in range(len(radius_list)):
            ref_list.append(utils.create_sphere(dim, radius_list[idx]))
        target_array = self.generate_with_shapes(objl, target_array, ref_list)
        return target_array


class Train(DeepFinder):
    def __init__(self, Ncl):
        DeepFinder.__init__(self)
        self.path_out = './'
        self.h5_dset_name = 'dataset' # if training set is stored as .h5 file, specify here in which h5 dataset the arrays are stored

        # Network parameters:
        self.Ncl = Ncl  # Ncl
        self.dim_in = 56  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.net = models.my_model(self.dim_in, self.Ncl)

        # Training parameters:
        self.batch_size = 25
        self.epochs = 100
        self.steps_per_epoch = 100
        self.Nvalid = 100  # number of samples for validation
        self.optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.flag_direct_read = 1
        self.flag_batch_bootstrap = 0
        self.Lrnd = 13  # random shifts applied when sampling data- and target-patches (in voxels)

    # This function launches the training procedure. For each epoch, an image is plotted, displaying the progression with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights are saved.
    # INPUTS:
    #   path_data     : a list containing the paths to data files (i.e. tomograms)
    #   path_target   : a list containing the paths to target files (i.e. annotated volumes)
    #   objlist_train : an xml structure containing information about annotated objects: origin tomogram (should correspond to the index of 'path_data' argument), coordinates, class. During training, these coordinates are used for guiding the patch sampling procedure.
    #   objlist_valid : same as 'objlist_train', but objects contained in this xml structure are not used for training, but for validation. It allows to monitor the training and check for over/under-fitting. Ideally, the validation objects should originate from different tomograms than training objects.
    # The network is trained on small 3D patches (i.e. sub-volumes), sampled from the larger tomograms (due to memory limitation). The patch sampling is not realized randomly, but is guided by the macromolecule coordinates contained in so-called object lists (objlist).
    # Concerning the loading of the dataset, two options are possible:
    #    flag_direct_read=0: the whole dataset is loaded into memory
    #    flag_direct_read=1: only the patches are loaded into memory, each time a training batch is generated. This is usefull when the dataset is too large to load into memory. However, the transfer speed between the data server and the GPU host should be high enough, else the procedure becomes very slow.
    def launch(self, path_data, path_target, objlist_train, objlist_valid):
        # Build network:
        self.net.compile(optimizer=self.optimizer, loss=losses.tversky_loss, metrics=['accuracy'])

        # Load whole dataset:
        if self.flag_direct_read == False:
            self.display('Loading dataset ...')
            data_list, target_list = core_utils.load_dataset(path_data, path_target, self.h5_dset_name)

        self.display('Launch training ...')

        # Declare lists for storing training statistics:
        hist_loss_train = []
        hist_acc_train  = []
        hist_loss_valid = []
        hist_acc_valid  = []
        hist_f1         = []
        hist_recall     = []
        hist_precision  = []
        process_time    = []

        # Training loop:
        for e in range(self.epochs):
            # TRAINING:
            start = time.time()
            for it in range(self.steps_per_epoch):
                if self.flag_direct_read:
                    batch_data, batch_target = self.generate_batch_direct_read(path_data, path_target, self.batch_size,
                                                                               objlist_train)
                else:
                    batch_data, batch_target = self.generate_batch_from_array(data_list, target_list, self.batch_size,
                                                                              objlist_train)
                loss_train = self.net.train_on_batch(batch_data, batch_target)

                self.display('epoch %d/%d - it %d/%d - loss: %0.3f - acc: %0.3f' % (
                e + 1, self.epochs, it + 1, self.steps_per_epoch, loss_train[0], loss_train[1]))
            hist_loss_train.append(loss_train[0])
            hist_acc_train.append(loss_train[1])

            # VALIDATION (compute statistics to monitor training):
            if self.flag_direct_read:
                batch_data_valid, batch_target_valid = self.generate_batch_direct_read(path_data, path_target,
                                                                                       self.batch_size, objlist_valid)
            else:
                batch_data_valid, batch_target_valid = self.generate_batch_from_array(data_list, target_list,
                                                                                      self.batch_size, objlist_valid)
            loss_val = self.net.evaluate(batch_data_valid, batch_target_valid, verbose=0)

            batch_pred = self.net.predict(batch_data_valid)
            scores = precision_recall_fscore_support(batch_target_valid.argmax(axis=-1).flatten(),
                                                     batch_pred.argmax(axis=-1).flatten(), average=None)

            hist_loss_valid.append(loss_val[0])
            hist_acc_valid.append(loss_val[1])
            hist_f1.append(scores[2])
            hist_recall.append(scores[1])
            hist_precision.append(scores[0])

            end = time.time()
            process_time.append(end - start)
            self.display('-------------------------------------------------------------')
            self.display('EPOCH %d/%d - valid loss: %0.3f - valid acc: %0.3f - %0.2fsec' % (
            e + 1, self.epochs, loss_val[0], loss_val[1], end - start))
            self.display('=============================================================')

            # Save and plot training history:
            history = {'loss': hist_loss_train, 'acc': hist_acc_train, 'val_loss': hist_loss_valid,
                       'val_acc': hist_acc_valid, 'val_f1': hist_f1, 'val_recall': hist_recall,
                       'val_precision': hist_precision}
            core_utils.save_history(history, self.path_out+'net_train_history.h5')
            core_utils.plot_history(history, self.path_out+'net_train_history_plot.png')

            if (e + 1) % 10 == 0:  # save weights every 10 epochs
                self.net.save(self.path_out+'net_weights_epoch' + str(e + 1) + '.h5')

        self.display("Model took %0.2f seconds to train" % np.sum(process_time))
        self.net.save(self.path_out+'net_weights_FINAL.h5')

    # This function generates training batches:
    #   - Data and target patches are sampled, in order to avoid loading whole tomograms.
    #   - The positions at which patches are sampled are determined by the coordinates contained in the object list.
    #   - Two data augmentation techniques are applied:
    #       .. To gain invariance to translations, small random shifts are added to the positions.
    #       .. 180 degree rotation around tilt axis (this way missing wedge orientation remains the same).
    #   - Also, bootstrap (i.e. re-sampling) can be applied so that we have an equal amount of each macromolecule in each batch.
    #     This is usefull when a class is under-represented.
    def generate_batch_direct_read(self, path_data, path_target, batch_size, objlist=None):
        p_in = np.int(np.floor(self.dim_in / 2))

        batch_data = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        # The batch is generated by randomly sampling data patches.
        if self.flag_batch_bootstrap:  # choose from bootstrapped objlist
            pool = core_utils.get_bootstrap_idx(objlist, Nbs=batch_size)
        else:  # choose from whole objlist
            pool = range(0, len(objlist))

        for i in range(batch_size):
            # Choose random object in training set:
            index = np.random.choice(pool)

            tomoID = int(objlist[index]['tomo_idx'])

            h5file = h5py.File(path_data[tomoID], 'r')
            tomodim = h5file['dataset'].shape  # get tomo dimensions without loading the array
            h5file.close()

            x, y, z = core_utils.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)

            # Load data and target patches:
            h5file = h5py.File(path_data[tomoID], 'r')
            patch_data = h5file['dataset'][x - p_in:x + p_in, y - p_in:y + p_in, z - p_in:z + p_in]
            h5file.close()

            h5file = h5py.File(path_target[tomoID], 'r')
            patch_target = h5file['dataset'][x - p_in:x + p_in, y - p_in:y + p_in, z - p_in:z + p_in]
            h5file.close()

            # Process the patches in order to be used by network:
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize
            patch_target_onehot = to_categorical(patch_target, self.Ncl)

            # Store into batch array:
            batch_data[i, :, :, :, 0] = patch_data
            batch_target[i] = patch_target_onehot

            # Data augmentation (180degree rotation around tilt axis):
            if np.random.uniform() < 0.5:
                batch_data[i] = np.rot90(batch_data[i], k=2, axes=(0, 2))
                batch_target[i] = np.rot90(batch_target[i], k=2, axes=(0, 2))

        return batch_data, batch_target

    def generate_batch_from_array(self, data, target, batch_size, objlist=None):
        p_in = np.int(np.floor(self.dim_in / 2))

        batch_data = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        # The batch is generated by randomly sampling data patches.
        if self.flag_batch_bootstrap:  # choose from bootstrapped objlist
            pool = core_utils.get_bootstrap_idx(objlist, Nbs=batch_size)
        else:  # choose from whole objlist
            pool = range(0, len(objlist))

        for i in range(batch_size):
            # choose random sample in training set:
            index = np.random.choice(pool)

            tomoID = int(objlist[index]['tomo_idx'])

            tomodim = data[tomoID].shape

            sample_data = data[tomoID]
            sample_target = target[tomoID]

            dim = sample_data.shape

            # Get patch position:
            x, y, z = core_utils.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)

            # Get patch:
            patch_data = sample_data[x - p_in:x + p_in, y - p_in:y + p_in, z - p_in:z + p_in]
            patch_batch = sample_target[x - p_in:x + p_in, y - p_in:y + p_in, z - p_in:z + p_in]

            # Process the patches in order to be used by network:
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize
            patch_batch_onehot = to_categorical(patch_batch, self.Ncl)

            # Store into batch array:
            batch_data[i, :, :, :, 0] = patch_data
            batch_target[i] = patch_batch_onehot

            # Data augmentation (180degree rotation around tilt axis):
            if np.random.uniform() < 0.5:
                batch_data[i] = np.rot90(batch_data[i], k=2, axes=(0, 2))
                batch_target[i] = np.rot90(batch_target[i], k=2, axes=(0, 2))

        return batch_data, batch_target


class Segment(DeepFinder):
    def __init__(self, Ncl, path_weights, patch_size=192):
        DeepFinder.__init__(self)

        self.Ncl = Ncl

        # Segmentation, parameters for dividing data in patches:
        self.P = patch_size  # patch length (in pixels) /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.pcrop = 25  # how many pixels to crop from border (net model dependent)
        self.poverlap = 55  # patch overlap (in pixels) (2*pcrop + 5)

        # Build network:
        self.net = models.my_model(self.P, self.Ncl)
        self.net.load_weights(path_weights)

    # This function enables to segment a tomogram. As tomograms are too large to be processed in one take, the tomogram is decomposed in smaller overlapping 3D patches.
    # INPUTS:
    #   dataArray: the volume to be segmented (3D numpy array)
    #   weights_path: path to the .h5 file containing the network weights obtained by the training procedure (string)
    # OUTPUT:
    #   predArray: a numpy array containing the predicted score maps.
    def launch(self, dataArray):
        dataArray = (dataArray[:] - np.mean(dataArray[:])) / np.std(dataArray[:])  # normalize
        dataArray = np.pad(dataArray, self.pcrop, mode='constant', constant_values=0)  # zeropad
        dim = dataArray.shape

        l = np.int(self.P / 2)
        lcrop = np.int(l - self.pcrop)
        step = np.int(2 * l + 1 - self.poverlap)

        # Get patch centers:
        pcenterX = list(range(l, dim[0] - l,
                              step))  # list() necessary for py3 (in py2 range() returns type 'list' but in py3 it returns type 'range')
        pcenterY = list(range(l, dim[1] - l, step))
        pcenterZ = list(range(l, dim[2] - l, step))

        # If there are still few pixels at the end:
        if pcenterX[-1] < dim[0] - l:
            pcenterX = pcenterX + [dim[0] - l, ]
        if pcenterY[-1] < dim[1] - l:
            pcenterY = pcenterY + [dim[1] - l, ]
        if pcenterZ[-1] < dim[2] - l:
            pcenterZ = pcenterZ + [dim[2] - l, ]

        Npatch = len(pcenterX) * len(pcenterY) * len(pcenterZ)
        self.display('Data array is divided in ' + str(Npatch) + ' patches ...')

        # ---------------------------------------------------------------
        # Process data in patches:
        start = time.time()

        predArray = np.zeros(dim + (self.Ncl,))
        normArray = np.zeros(dim)
        patchCount = 1
        for x in pcenterX:
            for y in pcenterY:
                for z in pcenterZ:
                    self.display('Segmenting patch ' + str(patchCount) + ' / ' + str(Npatch) + ' ...')
                    patch = dataArray[x - l:x + l, y - l:y + l, z - l:z + l]
                    patch = np.reshape(patch, (1, self.P, self.P, self.P, 1))  # reshape for keras [batch,x,y,z,channel]
                    pred = self.net.predict(patch, batch_size=1)

                    predArray[x - lcrop:x + lcrop, y - lcrop:y + lcrop, z - lcrop:z + lcrop, :] = predArray[
                                                                                                  x - lcrop:x + lcrop,
                                                                                                  y - lcrop:y + lcrop,
                                                                                                  z - lcrop:z + lcrop,
                                                                                                  :] + pred[0,
                                                                                                       l - lcrop:l + lcrop,
                                                                                                       l - lcrop:l + lcrop,
                                                                                                       l - lcrop:l + lcrop,
                                                                                                       :]
                    normArray[x - lcrop:x + lcrop, y - lcrop:y + lcrop, z - lcrop:z + lcrop] = normArray[
                                                                                               x - lcrop:x + lcrop,
                                                                                               y - lcrop:y + lcrop,
                                                                                               z - lcrop:z + lcrop] + np.ones(
                        (self.P - 2 * self.pcrop, self.P - 2 * self.pcrop, self.P - 2 * self.pcrop))

                    patchCount += 1

                    # Normalize overlaping regions:
        for C in range(0, self.Ncl):
            predArray[:, :, :, C] = predArray[:, :, :, C] / normArray

        end = time.time()
        self.display("Model took %0.2f seconds to predict" % (end - start))

        predArray = predArray[self.pcrop:-self.pcrop, self.pcrop:-self.pcrop, self.pcrop:-self.pcrop, :]  # unpad
        return predArray  # predArray is the array containing the scoremaps

    # Similar to function 'segment', only here the tomogram is not decomposed in smaller patches, but processed in one take. However, the tomogram array should be cubic, and the cube length should be a multiple of 4. This function has been developped for tests on synthetic data. I recommend using 'segment' rather than 'segment_single_block'.
    # INPUTS:
    #   dataArray: the volume to be segmented (3D numpy array)
    #   weights_path: path to the .h5 file containing the network weights obtained by the training procedure (string)
    # OUTPUT:
    #   predArray: a numpy array containing the predicted score maps.
    def launch_single_block(self, dataArray):
        dim = dataArray.shape
        dataArray = (dataArray[:] - np.mean(dataArray[:])) / np.std(dataArray[:])  # normalize
        dataArray = np.pad(dataArray, self.pcrop)  # zeropad
        dataArray = np.reshape(dataArray, (1, dim[0], dim[1], dim[2], 1))  # reshape for keras [batch,x,y,z,channel]

        pred = self.net.predict(dataArray, batch_size=1)
        predArray = pred[0, :, :, :, :]
        predArray = predArray[self.pcrop:-self.pcrop, self.pcrop:-self.pcrop, self.pcrop:-self.pcrop, :]  # unpad

        return predArray

class Cluster(DeepFinder):
    def __init__(self, clustRadius):
        DeepFinder.__init__(self)
        self.clustRadius = clustRadius
        self.sizeThr = 1

    # This function analyzes the segmented tomograms (i.e. labelmap), identifies individual macromolecules and outputs their coordinates. This is achieved with a clustering algorithm (meanshift).
    # INPUTS:
    #   labelmap: segmented tomogram (3D numpy array)
    #   sizeThr : cluster size (i.e. macromolecule size) (in voxels), under which a detected object is considered a false positive and is discarded
    #   clustRadius: parameter for clustering algorithm. Corresponds to average object radius (in voxels)
    # OUTPUT:
    #   objlist: a xml structure containing infos about detected objects: coordinates, class label and cluster size
    def launch(self, labelmap):
        Nclass = len(np.unique(labelmap)) - 1  # object classes only (background class not considered)

        objvoxels = np.nonzero(labelmap > 0)
        objvoxels = np.array(objvoxels).T  # convert to numpy array and transpose

        self.display('Launch clustering ...')
        start = time.time()
        clusters = MeanShift(bandwidth=self.clustRadius).fit(objvoxels)
        end = time.time()
        self.display("Clustering took %0.2f seconds" % (end - start))

        Nclust = clusters.cluster_centers_.shape[0]

        self.display('Analyzing clusters ...')
        objlist = []
        labelcount = np.zeros((Nclass,))
        for c in range(Nclust):
            clustMemberIndex = np.nonzero(clusters.labels_ == c)

            # Get cluster size and position:
            clustSize = np.size(clustMemberIndex)
            centroid = clusters.cluster_centers_[c]

            # Attribute a macromolecule class to cluster:
            clustMember = []
            for m in range(clustSize):  # get labels of cluster members
                clustMemberCoords = objvoxels[clustMemberIndex[0][m], :]
                clustMember.append(labelmap[clustMemberCoords[0], clustMemberCoords[1], clustMemberCoords[2]])

            for l in range(Nclass):  # get most present label in cluster
                labelcount[l] = np.size(np.nonzero(np.array(clustMember) == l + 1))
            winninglabel = np.argmax(labelcount) + 1

            objlist = ol.add_obj(objlist, label=winninglabel, coord=centroid, cluster_size=clustSize)

        self.display('Finished !')
        return objlist
