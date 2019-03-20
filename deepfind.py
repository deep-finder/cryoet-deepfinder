import h5py
import numpy as np
import time
import os

#from keras.utils import plot_model
from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import MeanShift

from lxml import etree

import models
import losses

class deepfind:
    def __init__(self, Ncl):
        # Network parameters:
        self.Ncl    = Ncl
        self.dim_in = 56 # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        
        # Training parameters:
        self.batch_size      = 25
        self.epochs          = 100
        self.steps_per_epoch = 100
        self.Nvalid          = 100 # number of samples for validation
        self.optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        self.Lrnd = 13 # random shifts applied when sampling data- and target-patches (in voxels)
        
        self.net = models.my_model(self.dim_in, self.Ncl)
        self.net.compile(optimizer=self.optimizer, loss=losses.tversky_loss, metrics=['accuracy'])
        
        # Segmentation, parameters for dividing data in patches:
        self.P        = 192 # patch length (in pixels) /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.poverlap = 70  # patch overlap (in pixels)
        self.pcrop    = 25  # how many pixels to crop from border
    
    # Parameters:
    #   path_data     : path to data (i.e. tomograms)
    #   path_target   : path to targets (i.e. annotated volumes)
    #   objlist_train : object list containing coordinates of macromolecules
    #   objlist_valid : another object list for monitoring the training (in particular detect overfitting)
    def train(self, path_data, path_target, objlist_train, objlist_valid):
        print('Launch training ...')
        
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
                batch_data, batch_target = self.generate_batch_direct_read(path_data, path_target, objlist_train, self.batch_size)
                loss_train = self.net.train_on_batch(batch_data, batch_target)
        
                print('epoch %d/%d - it %d/%d - loss: %0.3f - acc: %0.3f' % (e+1, self.epochs, it+1, self.steps_per_epoch, loss_train[0], loss_train[1]))
            hist_loss_train.append(loss_train[0])
            hist_acc_train.append( loss_train[1])
    
            # VALIDATION (compute statistics to monitor training):
            batch_data_valid, batch_target_valid = self.generate_batch_direct_read(path_data, path_target, objlist_valid, self.Nvalid)
            loss_val = self.net.evaluate(batch_data_valid, batch_target_valid, verbose=0)
    
            batch_pred = self.net.predict(batch_data_valid)
            scores = precision_recall_fscore_support(batch_target_valid.argmax(axis=-1).flatten(), batch_pred.argmax(axis=-1).flatten(), average=None)

            hist_loss_valid.append(loss_val[0])
            hist_acc_valid.append(loss_val[1])
            hist_f1.append(scores[2])
            hist_recall.append(scores[1])
            hist_precision.append(scores[0])
    
            end = time.time()
            process_time.append(end-start)
            print('-------------------------------------------------------------')
            print('EPOCH %d/%d - valid loss: %0.3f - valid acc: %0.3f - %0.2fsec' % (e+1, self.epochs, loss_val[0], loss_val[1], end-start))
            print('=============================================================')
    
            # Save training history:
            history = {'loss':hist_loss_train, 'acc':hist_acc_train, 'val_loss':hist_loss_valid, 'val_acc':hist_acc_valid, 'val_f1':hist_f1, 'val_recall':hist_recall, 'val_precision':hist_precision}
            self.save_history(history)
    
            if (e+1)%10 == 0: # save weights every 10 epochs
                self.net.save('params_model_epoch'+str(e+1)+'.h5')

        print "Model took %0.2f seconds to train"%np.sum(process_time)
        self.net.save('params_model_FINAL.h5')
        
        
    # This function generates training batches:
    #   - Data and target patches are sampled, in order to avoid loading whole tomograms.
    #   - The positions at which patches are sampled are determined by the coordinates contained in the object list.
    #   - Two data augmentation techniques are applied:
    #       .. To gain invariance to translations, small random shifts are added to the positions.
    #       .. 180 degree rotation around tilt axis (this way missing wedge orientation remains the same).
    #   - Also, bootstrap (i.e. re-sampling) is applied so that we have an equal amount of each macromolecule in each batch.
    #     This is usefull when a class is under-represented.
    
    # /!\ TO DO: add a test to ignore objlist coordinate too close to border (using tomodim=h5data['dataset'].shape)
    def generate_batch_direct_read(self, path_data, path_target, objlist, batch_size):
        Nobj = objlist.shape[0]
        p_in  = np.int( np.floor(self.dim_in /2) )
    
        batch_data   = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))
    
        lblTAB = np.unique(objlist[:,0])
        Nbs = 100
    
        # Bootstrap data so that we have equal frequencies (1/Nbs) for all classes:
        bs_idx = []
        for l in lblTAB:
            bs_idx.append( np.random.choice(np.squeeze(np.asarray(np.nonzero(objlist[:,0]==l))), Nbs) )
        bs_idx = np.concatenate(bs_idx)
        
        for i in range(batch_size):
            # Choose random sample in training set:
            index = np.random.choice(bs_idx)
            tomoID = objlist[index,1]
        
            # Add random shift to coordinates:
            x = objlist[index,4] + np.random.choice(range(-self.Lrnd,self.Lrnd+1))
            y = objlist[index,3] + np.random.choice(range(-self.Lrnd,self.Lrnd+1))
            z = objlist[index,2] + np.random.choice(range(-self.Lrnd,self.Lrnd+1))
        
            # Load data and target patches:
            h5data = h5py.File(path_data[tomoID], 'r')
            patch_data  = h5data['dataset'][x-p_in:x+p_in, y-p_in:y+p_in, z-p_in:z+p_in]
            patch_data  = (patch_data - np.mean(patch_data)) / np.std(patch_data) # normalize
            h5data.close()
            
            h5target = h5py.File(path_target[tomoID], 'r')
            patch_batch = h5target['dataset'][x-p_in:x+p_in, y-p_in:y+p_in, z-p_in:z+p_in]
            #patch_batch[patch_batch==-1] = 0 # /!\ -1 labels from 'ignore mask' could generate trouble
            patch_batch_onehot = to_categorical(patch_batch, self.Ncl)
            h5target.close()
            
            # Store into batch array:
            batch_data[i,:,:,:,0] = patch_data
            batch_target[i] = patch_batch_onehot
        
            # Data augmentation (180degree rotation around tilt axis):
            if np.random.uniform()<0.5:
                batch_data[i]   = np.rot90(batch_data[i]  , k=2, axes=(0,2))
                batch_target[i] = np.rot90(batch_target[i], k=2, axes=(0,2))
        
        return batch_data, batch_target
        
    def save_history(self, history):
        h5trainhist = h5py.File('params_train_history.h5', 'w')

        # train and val loss & accuracy:
        dset    = h5trainhist.create_dataset('acc', (len(history['acc']),))
        dset[:] = history['acc']
        dset    = h5trainhist.create_dataset('loss', (len(history['loss']),))
        dset[:] = history['loss']
        dset    = h5trainhist.create_dataset('val_acc', (len(history['val_acc']),))
        dset[:] = history['val_acc']
        dset    = h5trainhist.create_dataset('val_loss', (len(history['val_loss']),))
        dset[:] = history['val_loss']

        # val precision, recall, F1:
        dset    = h5trainhist.create_dataset('val_f1', np.shape(history['val_f1']))
        dset[:] = history['val_f1']
        dset    = h5trainhist.create_dataset('val_precision', np.shape(history['val_precision']))
        dset[:] = history['val_precision']
        dset    = h5trainhist.create_dataset('val_recall', np.shape(history['val_recall']))
        dset[:] = history['val_recall']

        h5trainhist.close()
        return
        
    def segment(self, dataArray, weights_path):
        self.net = models.my_model(self.P, self.Ncl)
        self.net.load_weights(weights_path)
        
        # Load data:
        #h5data    = h5py.File(path_data, 'r')
        #dataArray = h5data['dataset'][:]
        #h5data.close()
        dataArray = (dataArray[:] - np.mean(dataArray[:])) / np.std(dataArray[:]) # normalize
        dim  = dataArray.shape 
        
        l        = self.P/2
        lcrop    = l-self.pcrop
        step     = 2*l+1 - self.poverlap

        # Get patch centers:
        pcenterX = range(l, dim[0]-l, step)
        pcenterY = range(l, dim[1]-l, step)
        pcenterZ = range(l, dim[2]-l, step)

        # If there are still few pixels at the end:
        if pcenterX[-1]<dim[0]-l:
            pcenterX = pcenterX + [dim[0]-l,]
        if pcenterY[-1]<dim[1]-l:
            pcenterY = pcenterY + [dim[1]-l,]
        if pcenterZ[-1]<dim[2]-l:
            pcenterZ = pcenterZ + [dim[2]-l,]

        Npatch = len(pcenterX)*len(pcenterY)*len(pcenterZ)
        print('Data array is divided in ' + str(Npatch) + ' patches ...')

        #---------------------------------------------------------------
        # Process data in patches:
        start = time.time()

        predArray = np.zeros(dim+(self.Ncl,))
        normArray = np.zeros(dim)
        patchCount = 1
        for x in pcenterX:
            for y in pcenterY:
                for z in pcenterZ:
                    print('Segmenting patch ' + str(patchCount) + ' / ' + str(Npatch) + ' ...' )
                    patch = dataArray[x-l:x+l, y-l:y+l, z-l:z+l]
                    patch = np.reshape(patch, (1,self.P,self.P,self.P,1)) # reshape for keras [batch,x,y,z,channel]
                    pred = self.net.predict(patch, batch_size=1)
            
                    predArray[x-lcrop:x+lcrop, y-lcrop:y+lcrop, z-lcrop:z+lcrop, :] = predArray[x-lcrop:x+lcrop, y-lcrop:y+lcrop, z-lcrop:z+lcrop, :] + pred[0, l-lcrop:l+lcrop,l-lcrop:l+lcrop,l-lcrop:l+lcrop, :]
                    normArray[x-lcrop:x+lcrop, y-lcrop:y+lcrop, z-lcrop:z+lcrop]    = normArray[x-lcrop:x+lcrop, y-lcrop:y+lcrop, z-lcrop:z+lcrop] + np.ones((self.P-2*self.pcrop,self.P-2*self.pcrop,self.P-2*self.pcrop))
            
                    patchCount+=1  

        # Normalize overlaping regions:
        for C in range(0,self.Ncl):
            predArray[:,:,:,C] = predArray[:,:,:,C] / normArray

        end = time.time()
        print "Model took %0.2f seconds to predict"%(end - start)
        
        return predArray # predArray is the array containing the scoremaps
    
        # Save scoremaps:
        #path, filename = os.path.split(path_data)
        #scoremap_file = filename[:-3]+'_scoremaps.h5'
        #h5scoremap = h5py.File(scoremap_file, 'w')
        #for cl in range(0,self.Ncl):
    	#    dset = h5scoremap.create_dataset('class'+str(cl), (dim[0], dim[1], dim[2]), dtype='float16' )
    	#    dset[:] = np.float16(predArray[:,:,:,cl])
        #h5scoremap.close()
        
        # For binning: skimage.measure.block_reduce(mat, (2,2), np.mean)
        
    def cluster(self, labelmap, sizeThr, clustRadius):
        Nclass = len(np.unique(labelmap)) - 1 # object classes only (background class not considered)

        objvoxels = np.nonzero(labelmap>0)
        objvoxels = np.array(objvoxels).T # convert to numpy array and transpose

        print('Launch clustering ...')
        start = time.time()
        clusters = MeanShift(bandwidth=clustRadius).fit(objvoxels)
        end = time.time()
        print "Clustering took %0.2f seconds"%(end - start)

        Nclust = clusters.cluster_centers_.shape[0]

        print('Analyzing clusters ...')
        #objlist    = np.zeros((Nclust,5))
        objlist    = etree.Element('objlist')
        labelcount = np.zeros((Nclass,))
        for c in range(0,Nclust):
            clustMemberIndex = np.nonzero(clusters.labels_==c)
    
            # Get cluster size and position:
            clustSize = np.size(clustMemberIndex)
            centroid  = clusters.cluster_centers_[c]

            # Attribute a macromolecule class to cluster:
            clustMember = []
            for m in range(0,clustSize): # get labels of cluster members
                clustMemberCoords = objvoxels[clustMemberIndex[0][m],:]
                clustMember.append(labelmap[clustMemberCoords[0],clustMemberCoords[1],clustMemberCoords[2]])
        
            for l in range(0,Nclass): # get most present label in cluster
                labelcount[l] = np.size(np.nonzero( np.array(clustMember)==l+1 ))
            winninglabel = np.argmax(labelcount)+1
        
            # Store cluster infos in array:
            #objlist[c,0] = clustSize
            #objlist[c,1] = centroid[0]
            #objlist[c,2] = centroid[1]
            #objlist[c,3] = centroid[2]
            #objlist[c,4] = winninglabel
            
            obj = etree.SubElement(objlist, 'object')
            obj.set('cluster_size', str(clustSize))
            obj.set('class_label' , str(winninglabel))
            obj.set('x'           , '%.3f' % centroid[0])
            obj.set('y'           , '%.3f' % centroid[1])
            obj.set('z'           , '%.3f' % centroid[2])
        
        return objlist