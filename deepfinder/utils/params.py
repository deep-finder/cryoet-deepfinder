# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (Serpico team)
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

import numpy as np
from lxml import etree

from . import common as cm

class ParamsGenTarget():
    def __init__(self):
        self.path_objl = None # str
        self.path_initial_vol = None # str
        self.strategy = None # str
        self.radius_list = [None] # list of int
        self.path_mask_list = [None] # list of str

    def check(self):
        if type(self.path_objl)!=str:
            print('DeepFinder message: path_objl needs to be of type str.')
        if type(self.path_initial_vol)!=str:
            print('DeepFinder message: path_initial_vol needs to be of type str.')
        if type(self.strategy)!=str:
            print('DeepFinder message: strategy needs to be of type str.')
        if self.strategy!='spheres' and self.strategy!='shapes':
            print('DeepFinder message: strategy can only be "spheres" or "shapes".')
        for r in self.radius_list:
            if type(r)!=int:
                print('DeepFinder message: radius_list must contain only integers.')
        for p in self.path_mask_list:
            if type(p)!=str:
                print('DeepFinder message: path_mask_list must contain only strings.')

    def write(self, filename):
        root = etree.Element('paramsGenerateTarget')

        p = etree.SubElement(root, 'path_objl')
        p.set('path', str(self.path_objl))

        p = etree.SubElement(root, 'path_initial_vol')
        p.set('path', str(self.path_initial_vol))

        p = etree.SubElement(root, 'strategy')
        p.set('strategy', str(self.strategy))

        p = etree.SubElement(root, 'radius_list')
        for idx in range(len(self.radius_list)):
            pp = etree.SubElement(p, 'class' + str(idx + 1))
            pp.set('radius', str(self.radius_list[idx]))

        p = etree.SubElement(root, 'path_mask_list')
        for idx in range(len(self.path_mask_list)):
            pp = etree.SubElement(p, 'class'+str(idx + 1))
            pp.set('path', str(self.path_mask_list[idx]))

        tree = etree.ElementTree(root)
        tree.write(filename, pretty_print=True)

    def read(self, filename):
        tree = etree.parse(filename)
        root = tree.getroot()

        self.path_objl = root.find('path_objl').get('path')
        self.path_initial_vol = root.find('path_initial_vol').get('path')
        self.strategy = root.find('strategy').get('strategy')

        self.radius_list = []
        for idx in range(len(root.find('radius_list'))):
            radius = root.find('radius_list').find('class'+str(idx+1)).get('radius')
            self.radius_list.append( int(radius) )

        self.path_mask_list = []
        for idx in range(len(root.find('path_mask_list'))):
            path = root.find('path_mask_list').find('class' + str(idx + 1)).get('path')
            self.path_mask_list.append(path)

