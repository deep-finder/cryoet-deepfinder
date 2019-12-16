import numpy as np
import h5py
from skimage.measure import block_reduce
from lxml import etree
from copy import deepcopy
from sklearn.metrics import pairwise_distances


class objlist:
    # Constructors:
    def __init__(self, objlist=etree.Element('objlist')):
        self.objlist = objlist
        
    @classmethod
    def from_xml(cls, filename):
        tree = etree.parse(filename)
        return cls(objlist=tree.getroot())
        
    @classmethod
    def from_txt(cls, filename):
        #predicted_particles = []
        objlistOUT = objlist()
        with open(str(filename), 'rU') as f:
            for line in f:
                label, z, y, x, *_ = line.rstrip('\n').split()
                #predicted_particles.append((pdb, int(round(float(Z))), int(round(float(Y))), int(round(float(X)))))
                objlistOUT.add_obj(label, (float(x),float(y),float(z)))
        return objlistOUT

    # Methods:
    def add_obj(self, label, coord, cluster_size=None):
        obj = etree.SubElement(self.objlist, 'object')
        obj.set('class_label' , str(label))
        obj.set('x'           , '%.3f' % coord[0])
        obj.set('y'           , '%.3f' % coord[1])
        obj.set('z'           , '%.3f' % coord[2])
        if cluster_size!=None:
            obj.set('cluster_size', str(cluster_size))
            
    def size(self):
        return len(self.objlist)
        
    def read_xml(self, filename):
        tree = etree.parse(filename)
        self.objlist = tree.getroot()

    def write_xml(self, filename):
        tree = etree.ElementTree(self.objlist)
        tree.write(filename, pretty_print=True)

    def printhouba(self):
        print(etree.tostring(self.objlist))
        
    def get_class(self, label):
        N = len(self.objlist)
        label_list = np.zeros((N,))
        for idx in range(0,N):
            label_list[idx] = self.objlist[idx].get('class_label')
        idx_class = np.nonzero(label_list==label)
        idx_class = idx_class[0]
    
        objlistOUT = etree.Element('objlist')
        for idx in range(0,len(idx_class)):
            objlistOUT.append( deepcopy(self.objlist[idx_class[idx]]) ) # deepcopy is necessary, else the object is removed from objlIN when appended to objlOUT
        return objlist(objlist=objlistOUT)
        
    def above_thr(self, thr):
        N = len(self.objlist)
        clust_size_list = np.zeros((N,))
        for idx in range(0,N):
            if self.objlist[idx].get('cluster_size') != None:
                clust_size_list[idx] = self.objlist[idx].get('cluster_size')
            else:
                print('/!\ Object '+str(idx)+' has no attribute cluster_size')
        idx_thr = np.nonzero(clust_size_list>=thr)
        idx_thr = idx_thr[0]
    
        objlistOUT = etree.Element('objlist')
        for idx in range(0,len(idx_thr)):
            objlistOUT.append( deepcopy(self.objlist[idx_thr[idx]]) ) # deepcopy is necessary, else the object is removed from objlIN when appended to objlOUT
        return objlist(objlist=objlistOUT)
    
    # TODO check why np.round is used
    def scale_coord(self, scale):
        objlistOUT = self.objlist 
        for p in range(0,len(self.objlist)):
            x = int(np.round(float( self.objlist[p].get('x') )))
            y = int(np.round(float( self.objlist[p].get('y') )))
            z = int(np.round(float( self.objlist[p].get('z') )))
            x = scale*x
            y = scale*y
            z = scale*z
            objlistOUT[p].set('x', str(x))
            objlistOUT[p].set('y', str(y))
            objlistOUT[p].set('z', str(z))
        return objlist(objlist=objlistOUT)
    
    # /!\ for now this function does not know how to handle empty objlists
    def get_Ntp(self, objl_gt, tol_pos_err):
        Ngt = objl_gt.size()
        Ndf = len(self.objlist)
        coords_gt = np.zeros((Ngt,3))
        coords_df = np.zeros((Ndf,3))

        for idx in range(0,Ngt):
            coords_gt[idx,0] = objl_gt.objlist[idx].get('x')
            coords_gt[idx,1] = objl_gt.objlist[idx].get('y')
            coords_gt[idx,2] = objl_gt.objlist[idx].get('z')
        for idx in range(0,Ndf):
            coords_df[idx,0] = self.objlist[idx].get('x')
            coords_df[idx,1] = self.objlist[idx].get('y')
            coords_df[idx,2] = self.objlist[idx].get('z')

        # Get pairwise distance matrix:
        D = pairwise_distances(coords_gt, coords_df, metric='euclidean')

        # Get pairs that are closer than tol_pos_err:
        D = D<=tol_pos_err

        # A detected object is considered a true positive (TP) if it is closer than tol_pos_err to a ground truth object.
        match_vector = np.sum(D,axis=1)
        Ntp = np.sum(match_vector==1)
        return Ntp
        
        
