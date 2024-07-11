# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import numpy as np
import os, time, h5py

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from deepfinder.models import model_loader
from deepfinder import callbacks, losses
from deepfinder.utils import core

# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# TODO: add method for resuming training. It should load existing weights and train_history. So when restarting, the plot curves show prececedent epochs
class Train(core.DeepFinder):
    def __init__(self, Ncl, dim_in):
        core.DeepFinder.__init__(self)
        self.path_out = './'
        self.h5_dset_name = 'dataset' # if training set is stored as .h5 file, specify here in which h5 dataset the arrays are stored

        # Network parameters:
        self.Ncl = Ncl  # Ncl
        self.dim_in = dim_in  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.net = None

        self.label_list = []
        for l in range(self.Ncl): self.label_list.append(l) # for precision_recall_fscore_support
                                                            # (else bug if not all labels exist in batch)

        # Training parameters:
        self.batch_size = 25
        self.epochs = 100
        self.steps_per_epoch = 100
        self.steps_per_valid = 10  # number of samples for validation
        self.optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.loss = losses.tversky_loss

        self.flag_direct_read = 1
        self.flag_batch_bootstrap = 0
        
        # random shifts applied when sampling data- and target-patches (in voxels)
        self.Lrnd = 13 

        self.class_weight = None
        self.sample_weights = None  # np array same lenght as objl_train

        self.check_attributes()

    def check_attributes(self):
        self.is_positive_int(self.Ncl, 'Ncl')
        self.is_multiple_4_int(self.dim_in, 'dim_in')
        self.is_positive_int(self.batch_size, 'batch_size')
        self.is_positive_int(self.epochs, 'epochs')
        self.is_positive_int(self.steps_per_epoch, 'steps_per_epoch')
        self.is_positive_int(self.steps_per_valid, 'steps_per_valid')
        self.is_int(self.Lrnd, 'Lrnd')

    # This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
    # with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
    # are saved.
    # INPUTS:
    #   path_data     : a list containing the paths to data files (i.e. tomograms)
    #   path_target   : a list containing the paths to target files (i.e. annotated volumes)
    #   objlist_train : list of dictionaries containing information about annotated objects (e.g. class, position)
    #                   In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
    #                   See utils/objl.py for more info about object lists.
    #                   During training, these coordinates are used for guiding the patch sampling procedure.
    #   objlist_valid : same as 'objlist_train', but objects contained in this list are not used for training,
    #                   but for validation. It allows to monitor the training and check for over/under-fitting. Ideally,
    #                   the validation objects should originate from different tomograms than training objects.
    # The network is trained on small 3D patches (i.e. sub-volumes), sampled from the larger tomograms (due to memory
    # limitation). The patch sampling is not realized randomly, but is guided by the macromolecule coordinates contained
    # in so-called object lists (objlist).
    # Concerning the loading of the dataset, two options are possible:
    #    flag_direct_read=0: the whole dataset is loaded into memory
    #    flag_direct_read=1: only the patches are loaded into memory, each time a training batch is generated. This is
    #                        usefull when the dataset is too large to load into memory. However, the transfer speed
    #                        between the data server and the GPU host should be high enough, else the procedure becomes
    #                        very slow.
    # TODO: delete flag_direct_read. Launch should detect if direct_read is desired by checking if input data_list and
    #       target_list contain str (path) or numpy array
    def launch(self, path_data, path_target, objlist_train, objlist_valid):
        """This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
        with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
        are saved.

        Args:
            path_data (list of string): contains paths to data files (i.e. tomograms)
            path_target (list of string): contains paths to target files (i.e. annotated volumes)
            objlist_train (list of dictionaries): contains information about annotated objects (e.g. class, position)
                In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
                See utils/objl.py for more info about object lists.
                During training, these coordinates are used for guiding the patch sampling procedure.
            objlist_valid (list of dictionaries): same as 'objlist_train', but objects contained in this list are not
                used for training, but for validation. It allows to monitor the training and check for over/under-fitting.
                Ideally, the validation objects should originate from different tomograms than training objects.

        Note:
            The function saves following files at regular intervals:
                net_weights_epoch*.h5: contains current network weights

                net_train_history.h5: contains arrays with all metrics per training iteration

                net_train_history_plot.png: plotted metric curves

        """
        self.check_attributes()
        self.check_arguments(path_data, path_target, objlist_train, objlist_valid)

        if self.net is None:
            self.net = model_loader.load_model(self.dim_in, self.Ncl, 'unet', None)    

        # TensorBoard writer
        log_dir = self.path_out + "tensorboard_logs/"
        writer = tf.summary.create_file_writer(log_dir)        
        tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='500,520')

        # gpus = tf.config.list_logical_devices('GPU')
        # strategy = tf.distribute.MirroredStrategy(gpus)

        # Check GPU memory limit
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Set the Desired GPU if Multi-Model Training is Specified
                if self.gpuID is not None:
                    tf.config.experimental.set_visible_devices(gpus[self.gpuID], "GPU")
                # Enable memory growth for the GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found.")

        # Build network (not in constructor, else not possible to init model with weights from previous train round):
        self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        self.batch_data = np.zeros((self.batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        self.batch_target = np.zeros((self.batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        train_dataset = self.create_tf_dataset(data_list, target_list, objlist_train, self.batch_size, self.dim_in, self.Ncl, self.flag_batch_bootstrap)
        valid_dataset = self.create_tf_dataset(data_list, target_list, objlist_valid, self.batch_size, self.dim_in, self.Ncl, self.flag_batch_bootstrap)

        # Callbacks for Save weights and Clear Memory
        # clear_memory_callback = callbacks.ClearMemoryCallback()        
        save_weights_callback = callbacks.SaveWeightsCallback(self.path_out)
        learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

        # Load whole dataset:
        if self.flag_direct_read == False:
            self.display('Loading dataset ...')
            data_list, target_list = core.load_dataset(path_data, path_target, self.h5_dset_name)

        plotting_callback = callbacks.TrainingPlotCallback(validation_data=valid_dataset, validation_steps=self.steps_per_valid, path_out=self.path_out, label_list=self.label_list)

        self.display('Launch training ...')

        # Train the model using model.fit()
        history = self.net.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            class_weight=self.class_weights,
            validation_data=valid_dataset,
            validation_steps=self.steps_per_valid,
            callbacks=[tf_callback, save_weights_callback,plotting_callback,learning_rate_callback],
            verbose=1
        )

        # self.display("Model took %0.2f seconds to train" % np.sum(process_time))
        self.net.save(self.path_out+'net_weights_FINAL.h5')

    def check_arguments(self, path_data, path_target, objlist_train, objlist_valid):
        self.is_list(path_data, 'path_data')
        self.is_list(path_target, 'path_target')
        self.are_lists_same_length([path_data, path_target], ['data_list', 'target_list'])
        self.is_list(objlist_train, 'objlist_train')
        self.is_list(objlist_valid, 'objlist_valid')

    def create_tf_dataset(self, data, target, objlist, batch_size, dim_in, Ncl, flag_batch_bootstrap):
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(data, target, objlist, batch_size, dim_in, Ncl, flag_batch_bootstrap),
            output_signature=(
                tf.TensorSpec(shape=(batch_size, dim_in, dim_in, dim_in, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, dim_in, dim_in, dim_in, Ncl), dtype=tf.float32)
            )
        )
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def data_generator(self,data, target, objlist, batch_size, dim_in, Ncl, flag_batch_bootstrap):
        p_in = int(np.floor(dim_in / 2))

        while True:
            
            if flag_batch_bootstrap:
                pool = core.get_bootstrap_idx(objlist, Nbs=batch_size)
            else:
                pool = range(0, len(objlist))

            idx_list = []
            for i in range(batch_size):
                index = np.random.choice(pool)
                idx_list.append(index)

                tomoID = int(objlist[index]['tomo_idx'])
                tomodim = data[tomoID].shape
                sample_data = data[tomoID]
                sample_target = target[tomoID]

                x, y, z = core.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)

                patch_data = sample_data[z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]
                patch_target = sample_target[z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]

                patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)
                patch_target_onehot = to_categorical(patch_target, Ncl)

                self.batch_data[i, :, :, :, 0] = patch_data
                self.batch_target[i] = patch_target_onehot

                if np.random.uniform() < 0.5:
                    self.batch_data[i] = np.rot90(self.batch_data[i], k=2, axes=(0, 2))
                    self.batch_target[i] = np.rot90(self.batch_target[i], k=2, axes=(0, 2))

            yield self.batch_data, self.batch_target


