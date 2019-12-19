import numpy as np
import h5py
from skimage.measure import block_reduce
from lxml import etree
from copy import deepcopy
from sklearn.metrics import pairwise_distances

import utils

class objlist: # here I want to test a list of dictionnaries as core structure and see if its faster
    # Constructors:
    def __init__(self, objlist=None):
        if objlist is None: # this syntax is necessary, else objects share the same attributes
            objlist = []
        self.objlist = objlist

    # TODO: handle psi phi theta
    @classmethod
    def from_xml(cls, filename):
        tree = etree.parse(filename)
        objl_xml = tree.getroot()

        objlOUT = objlist()
        for p in range(len(objl_xml)):
            lbl = objl_xml[p].get('class_label')
            x   = objl_xml[p].get('x')
            y   = objl_xml[p].get('y')
            z   = objl_xml[p].get('z')
            cluster_size = objl_xml[p].get('cluster_size')
            objlOUT.add_obj(label=lbl, coord=(float(x), float(y), float(z)), cluster_size=int(cluster_size))
        return objlOUT

    # TODO: handle psi phi theta
    @classmethod
    def from_txt(cls, filename):
        objlOUT = objlist()
        with open(str(filename), 'rU') as f:
            for line in f:
                lbl, z, y, x, *_ = line.rstrip('\n').split()
                objlOUT.add_obj(label=lbl, coord=(float(x), float(y), float(z)))
        return objlOUT

    # Methods:
    def add_obj(self, label, coord, orient=(None,None,None), cluster_size=None):
        obj = {
            'label': label,
            'x'    :coord[0] ,
            'y'    :coord[1] ,
            'z'    :coord[2] ,
            'psi'  :orient[0],
            'phi'  :orient[1],
            'the'  :orient[2],
            'cluster_size':cluster_size
        }
        self.objlist.append(obj)

    def size(self):
        return len(self.objlist)

    def print(self):
        for p in range(len(self.objlist)):
            lbl = self.objlist[p]['label']
            x = self.objlist[p]['x']
            y = self.objlist[p]['y']
            z = self.objlist[p]['z']
            psi = self.objlist[p]['psi']
            phi = self.objlist[p]['phi']
            the = self.objlist[p]['the']
            cluster_size = self.objlist[p]['cluster_size']

            if cluster_size==None:
                print('obj ' + str(p) + ': (lbl:' + str(lbl) + ', x:' + str(x) + ', y:'+str(y) + ', z:' + str(z) + ')')
            else:
                print('obj ' + str(p) + ': (lbl:' + str(lbl) + ', x:' + str(x) + ', y:' + str(y) + ', z:' + str(z) + ', cluster_size:'+str(cluster_size)+')')

    # label can be int or str (is casted to str)
    def get_class(self, label):
        N = len(self.objlist)
        idx_class = []
        for idx in range(N):
            if self.objlist[idx]['label']==str(label):
                idx_class.append(idx)

        objlistOUT = []
        for idx in range(len(idx_class)):
            objlistOUT.append(self.objlist[idx_class[idx]])
        return objlist(objlist=objlistOUT)

    def above_thr(self, thr):
        N = len(self.objlist)
        idx_thr = []
        for idx in range(N):
            csize = self.objlist[idx]['cluster_size']
            if csize != None:
                if csize>=thr:
                    idx_thr.append(idx)
            else:
                print('/!\ Object ' + str(idx) + ' has no attribute cluster_size')

        objlistOUT = []
        for idx in range(0,len(idx_thr)):
            objlistOUT.append( self.objlist[idx_thr[idx]] )
        return objlist(objlist=objlistOUT)

    def write_xml(self, filename):
        objl_xml = etree.Element('objlist')
        for idx in range(len(self.objlist)):
            lbl   = self.objlist[idx]['label']
            x     = self.objlist[idx]['x']
            y     = self.objlist[idx]['y']
            z     = self.objlist[idx]['z']
            csize = self.objlist[idx]['cluster_size']

            obj = etree.SubElement(objl_xml, 'object')
            obj.set('class_label' , str(lbl))
            obj.set('x'           , '%.3f' % x)
            obj.set('y'           , '%.3f' % y)
            obj.set('z'           , '%.3f' % z)
            if csize!=None:
                obj.set('cluster_size', str(csize))

        tree = etree.ElementTree(objl_xml)
        tree.write(filename, pretty_print=True)





# class objlist:
#     # Constructors:
#     def __init__(self, objlist=etree.Element('objlist')):
#         self.objlist = objlist
#
#     @classmethod
#     def from_xml(cls, filename):
#         tree = etree.parse(filename)
#         return cls(objlist=tree.getroot())
#
#     @classmethod
#     def from_txt(cls, filename):
#         #predicted_particles = []
#         objlistOUT = objlist()
#         with open(str(filename), 'rU') as f:
#             for line in f:
#                 label, z, y, x, *_ = line.rstrip('\n').split()
#                 #predicted_particles.append((pdb, int(round(float(Z))), int(round(float(Y))), int(round(float(X)))))
#                 objlistOUT.add_obj(label, (float(x),float(y),float(z)))
#         return objlistOUT
#
#     # Methods:
#     def add_obj(self, label, coord, cluster_size=None):
#         obj = etree.SubElement(self.objlist, 'object')
#         obj.set('class_label' , str(label))
#         obj.set('x'           , '%.3f' % coord[0])
#         obj.set('y'           , '%.3f' % coord[1])
#         obj.set('z'           , '%.3f' % coord[2])
#         if cluster_size!=None:
#             obj.set('cluster_size', str(cluster_size))
#
#     def size(self):
#         return len(self.objlist)
#
#     def read_xml(self, filename):
#         tree = etree.parse(filename)
#         self.objlist = tree.getroot()
#
#     def write_xml(self, filename):
#         tree = etree.ElementTree(self.objlist)
#         tree.write(filename, pretty_print=True)
#
#     def printhouba(self):
#         print(etree.tostring(self.objlist))
#
#     def get_class(self, label):
#         N = len(self.objlist)
#         label_list = np.zeros((N,))
#         for idx in range(0,N):
#             label_list[idx] = self.objlist[idx].get('class_label')
#         idx_class = np.nonzero(label_list==label)
#         idx_class = idx_class[0]
#
#         objlistOUT = etree.Element('objlist')
#         for idx in range(0,len(idx_class)):
#             objlistOUT.append( deepcopy(self.objlist[idx_class[idx]]) ) # deepcopy is necessary, else the object is removed from objlIN when appended to objlOUT
#         return objlist(objlist=objlistOUT)
#
#     def above_thr(self, thr):
#         N = len(self.objlist)
#         clust_size_list = np.zeros((N,))
#         for idx in range(0,N):
#             if self.objlist[idx].get('cluster_size') != None:
#                 clust_size_list[idx] = self.objlist[idx].get('cluster_size')
#             else:
#                 print('/!\ Object '+str(idx)+' has no attribute cluster_size')
#         idx_thr = np.nonzero(clust_size_list>=thr)
#         idx_thr = idx_thr[0]
#
#         objlistOUT = etree.Element('objlist')
#         for idx in range(0,len(idx_thr)):
#             objlistOUT.append( deepcopy(self.objlist[idx_thr[idx]]) ) # deepcopy is necessary, else the object is removed from objlIN when appended to objlOUT
#         return objlist(objlist=objlistOUT)
#
#     # TODO check why np.round is used
#     def scale_coord(self, scale):
#         objlistOUT = self.objlist
#         for p in range(0,len(self.objlist)):
#             x = int(np.round(float( self.objlist[p].get('x') )))
#             y = int(np.round(float( self.objlist[p].get('y') )))
#             z = int(np.round(float( self.objlist[p].get('z') )))
#             x = scale*x
#             y = scale*y
#             z = scale*z
#             objlistOUT[p].set('x', str(x))
#             objlistOUT[p].set('y', str(y))
#             objlistOUT[p].set('z', str(z))
#         return objlist(objlist=objlistOUT)
#
#     # /!\ for now this function does not know how to handle empty objlists
#     def get_Ntp(self, objl_gt, tol_pos_err):
#         Ngt = objl_gt.size()
#         Ndf = len(self.objlist)
#         coords_gt = np.zeros((Ngt,3))
#         coords_df = np.zeros((Ndf,3))
#
#         for idx in range(0,Ngt):
#             coords_gt[idx,0] = objl_gt.objlist[idx].get('x')
#             coords_gt[idx,1] = objl_gt.objlist[idx].get('y')
#             coords_gt[idx,2] = objl_gt.objlist[idx].get('z')
#         for idx in range(0,Ndf):
#             coords_df[idx,0] = self.objlist[idx].get('x')
#             coords_df[idx,1] = self.objlist[idx].get('y')
#             coords_df[idx,2] = self.objlist[idx].get('z')
#
#         # Get pairwise distance matrix:
#         D = pairwise_distances(coords_gt, coords_df, metric='euclidean')
#
#         # Get pairs that are closer than tol_pos_err:
#         D = D<=tol_pos_err
#
#         # A detected object is considered a true positive (TP) if it is closer than tol_pos_err to a ground truth object.
#         match_vector = np.sum(D,axis=1)
#         Ntp = np.sum(match_vector==1)
#         return Ntp
        
        
class scoremaps:
    # Constructors:
    def __init__(self, array=None):
        self.array = array # numpy array of size (dim[0],dim[1],dim[2],Ncl)
        
    @classmethod
    def from_h5(cls, filename):
        h5file = h5py.File(filename, 'r')
        datasetnames = h5file.keys()
        Ncl = len(datasetnames)
        dim = h5file['class0'].shape 
        array = np.zeros((dim[0],dim[1],dim[2],Ncl))
        for cl in range(0,Ncl):
            array[:,:,:,cl] = h5file['class'+str(cl)][:]
        h5file.close()
        return cls(array=array)
        
    # to do
    # @classmethod
    # def from_mrc(cls, filename):
    #     return cls(array=array)
        
    # Methods:
    def write_h5(self, filename):
        h5file = h5py.File(filename, 'w')
        dim = self.array.shape
        Ncl = dim[3]
        for cl in range(0,Ncl):
    	    dset = h5file.create_dataset('class'+str(cl), (dim[0], dim[1], dim[2]), dtype='float16' )
    	    dset[:] = np.float16(self.array[:,:,:,cl])
        h5file.close()
        
    # def write_mrc(self, filename):
        
    def to_lblmap(self):
        lmap = np.argmax(self.array,3)
        return lblmap(array=lmap)
        
    def bin(self):
        dim = self.array.shape
        Ncl = dim[3]
        dimB = (int(np.round(dim[0]/2)), int(np.round(dim[1]/2)), int(np.round(dim[2]/2)), Ncl)
        arrayB = np.zeros(dimB)
        for cl in range(0,Ncl):
            arrayB[:,:,:,cl] = block_reduce(self.array[:,:,:,cl], (2,2,2), np.mean)
        return scoremaps(arrayB)


class lblmap:
    # Constructors:
    def __init__(self, array=None):
        self.array = array # numpy array of size (dim[0],dim[1],dim[2])

    @classmethod
    def from_h5(cls, filename):
        array = utils.read_h5array(filename)
        return cls(array=array)

    @classmethod
    def from_mrc(cls, filename):
        array = utils.read_mrc(filename)
        return cls(array=array)

    # Methods:
    def write_h5(self, filename):
        dim = self.array.shape
        h5file = h5py.File(filename, 'w')
        dset = h5file.create_dataset('dataset', (dim[0], dim[1], dim[2]), dtype='int8')
        dset[:] = np.int8(self.array)
        h5file.close()

    def write_mrc(self, filename):
        utils.write_mrc(self.array, filename)

