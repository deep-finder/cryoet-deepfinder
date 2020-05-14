# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (Serpico team)
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox
import pyqtgraph as pg
import numpy as np
import matplotlib # for lmap colormap

import sys
sys.path.append('../')
from custom_theme import display_message_box

class DisplayOrthoslicesWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.gl = pg.GraphicsLayoutWidget()

        # This is necessary to attach the GraphicsLayoutWidget to this DisplayOrthoslicesWidget:
        self.layout = QVBoxLayout()
        QWidget.setLayout(self, self.layout)
        self.layout.addWidget(self.gl)

        # Orthoslices:
        self.vb_xy = CustomViewBox(invertY=True)
        self.vb_zy = CustomViewBox(invertY=True)
        self.vb_zx = CustomViewBox(invertY=True)
        self.gl.addItem(self.vb_xy, row=0, col=0)
        self.gl.addItem(self.vb_zy, row=0, col=1)
        self.gl.addItem(self.vb_zx, row=1, col=0)

        self.vb_xy.setAspectLocked(lock=True, ratio=1) # lock the aspect ratio so pixels are always square
        self.vb_zy.setAspectLocked(lock=True, ratio=1)
        self.vb_zx.setAspectLocked(lock=True, ratio=1)

        # self.vb_xy.setMouseEnabled(x=False, y=False) # disable move image and zoom with mouse
        # self.vb_zx.setMouseEnabled(x=False, y=False)
        # self.vb_zy.setMouseEnabled(x=False, y=False)

        self.img_xy = pg.ImageItem() # tomogram slices
        self.img_zy = pg.ImageItem()
        self.img_zx = pg.ImageItem()

        self.vb_xy.addItem(self.img_xy) # link tomogram slices to viewboxes
        self.vb_zy.addItem(self.img_zy)
        self.vb_zx.addItem(self.img_zx)

        self.img_lmap_xy = pg.ImageItem() # label map slices
        self.img_lmap_zy = pg.ImageItem()
        self.img_lmap_zx = pg.ImageItem()

        self.vb_xy.addItem(self.img_lmap_xy) # link lmap slices to viewboxes
        self.vb_zy.addItem(self.img_lmap_zy)
        self.vb_zx.addItem(self.img_lmap_zx)

        # Cursor lines:
        self.lineV_xy = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.lineH_xy = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.vb_xy.addItem(self.lineV_xy)
        self.vb_xy.addItem(self.lineH_xy)

        self.lineV_zy = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.lineH_zy = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.vb_zy.addItem(self.lineV_zy)
        self.vb_zy.addItem(self.lineH_zy)

        self.lineV_zx = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.lineH_zx = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.vb_zx.addItem(self.lineV_zx)
        self.vb_zx.addItem(self.lineH_zx)

        # Label displaying coordinates:
        self.label = pg.LabelItem()
        self.gl.addItem(self.label, row=1, col=1)

        # Relative to displayed tomogram, needs to be initialized by set_vol()
        self.vol = None
        self.dim = (None, None, None)
        self.levels = (None,None)
        self.vol_min = None
        self.vol_max = None
        self.vol_mu = None
        self.vol_sig = None
        self.x = None
        self.y = None
        self.z = None
        self.isTomoLoaded = False

        # Relative to displayed label map, needs to be initialized by set_lmap()
        self.lmap = None
        self.levels_lmap = (0, 4) # supposed max nb of classes TODO: adapt if >25
        self.isLmapLoaded = False

        # Connect click signal to dedicated function:
        self.gl.scene().sigMouseClicked.connect(self.mouseClick_secure)

        # Link axis of orthoslices:
        self.vb_xy.sigRangeChangedManually.connect(self.link_axes_vb_xy)
        self.vb_zx.sigRangeChangedManually.connect(self.link_axes_vb_zx)
        self.vb_zy.sigRangeChangedManually.connect(self.link_axes_vb_zy)

    def link_axes_vb_xy(self):
        self.vb_zx.setXRange(*self.vb_xy.viewRange()[0])
        self.vb_zy.setYRange(*self.vb_xy.viewRange()[1])

    def link_axes_vb_zx(self):
        self.vb_xy.setXRange(*self.vb_zx.viewRange()[0])
        self.vb_zy.setXRange(*self.vb_zx.viewRange()[1])

    def link_axes_vb_zy(self):
        self.vb_zx.setYRange(*self.vb_zy.viewRange()[0])
        self.vb_xy.setYRange(*self.vb_zy.viewRange()[1])

    def get_orthoslices(self, volume):
        slice_xy = np.transpose(volume[self.z, :, :])
        slice_zx = np.transpose(volume[:, self.y, :])
        slice_zy = volume[:, :, self.x]
        return slice_xy, slice_zx, slice_zy

    def set_vol(self, vol):
        self.isTomoLoaded = True
        self.vol = vol
        self.dim = self.vol.shape
        self.vol_mu = np.mean(vol)
        self.vol_sig = np.std(vol)
        self.levels = (self.vol_mu-5*self.vol_sig,self.vol_mu+5*self.vol_sig)
        self.vol_min = np.min(vol)
        self.vol_max = np.max(vol)

        self.initialize_display() # needs to be called before get_orthoslices, because instanciates x,y,z

        slice_xy, slice_zx, slice_zy = self.get_orthoslices(self.vol)
        self.img_xy.setImage(slice_xy, levels=self.levels)
        self.img_zx.setImage(slice_zx, levels=self.levels)
        self.img_zy.setImage(slice_zy, levels=self.levels)

        self.initialize_orthoslice_linking()

    def initialize_display(self): # /!\ self.dim needs to be declared first.
        # Stuff that needs to be intialized after loading a volume (tomogram or label map)
        self.x = np.int(np.round(self.dim[2] / 2))
        self.y = np.int(np.round(self.dim[1] / 2))
        self.z = np.int(np.round(self.dim[0] / 2))

        self.lineV_xy.setPos(self.x)
        self.lineH_xy.setPos(self.y)
        self.lineV_zy.setPos(self.z)
        self.lineH_zy.setPos(self.y)
        self.lineV_zx.setPos(self.x)
        self.lineH_zx.setPos(self.z)

        # Initialize zoom centers:
        self.set_zoom_centers()

        # Initialize coordinate display:
        self.label.setText('(x,y,z)=' + '(' + str(self.x) + ',' + str(self.y) + ',' + str(self.z) + ')')

    def initialize_orthoslice_linking(self): # has to be called after setImage
        # For some strange reason, this is necessary to initialize orthoslice linking for zoom and translation:
        # (initially I had to quickly zoom in/out each slice)
        self.vb_xy.scaleBy(1, (self.x, self.y))
        self.vb_zx.scaleBy(1, (self.x, self.z))
        self.vb_zy.scaleBy(1, (self.z, self.y))

    def set_vol_levels(self, levels):
        self.levels = levels
        self.img_xy.setLevels(levels)
        self.img_zx.setLevels(levels)
        self.img_zy.setLevels(levels)

    def set_lmap(self, lmap):
        self.isLmapLoaded = True
        self.lmap = lmap
        self.dim = self.lmap.shape

        self.initialize_display() # needs to be called before get_orthoslices, because instanciates x,y,z

        lmap_xy, lmap_zx, lmap_zy = self.get_orthoslices(lmap) # (x,y,z) need to be initialized first, else crash

        self.set_lmap_color_map()
        self.img_lmap_xy.setImage(lmap_xy, levels=self.levels_lmap)
        self.img_lmap_zy.setImage(lmap_zy, levels=self.levels_lmap)
        self.img_lmap_zx.setImage(lmap_zx, levels=self.levels_lmap)

        self.set_lmap_opacity(1)
        self.initialize_orthoslice_linking()

    def set_lmap_color_map(self):
        # Set color map:
        # colors = [
        #     (0, 0, 0),
        #     (45, 5, 61),
        #     (84, 42, 55),
        #     (150, 87, 60),
        #     (208, 171, 141),
        #     (255, 255, 255)
        # ]
        # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        # lut = cmap.getLookupTable(alpha=True)
        # # add alpha channel so that '0'->transparent
        # alpha = np.ones((lut.shape[0], 1)) * 255
        # alpha[0] = 0
        # lut = np.concatenate((lut, alpha), axis=1)

        colormap = matplotlib.cm.get_cmap('gist_ncar')  # CMRmap
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        lut = np.random.permutation(lut)  # so that adjacent classes (e.g. 1 & 2) colors are not too similar
        # edit alpha channel so that '0'->transparent
        alpha = np.ones(lut.shape[0]) * 255
        alpha[0] = 0
        lut[:, 3] = alpha

        self.img_lmap_xy.setLookupTable(lut)
        self.img_lmap_zy.setLookupTable(lut)
        self.img_lmap_zx.setLookupTable(lut)

    def set_lmap_opacity(self, opacity):
        self.img_lmap_xy.setOpacity(opacity)
        self.img_lmap_zy.setOpacity(opacity)
        self.img_lmap_zx.setOpacity(opacity)

    # For interactive lmap creation in annotation tool:
    def update_lmap(self, lmap):
        self.isLmapLoaded = True
        self.lmap = lmap

    def goto_coord(self, coord=[None,None,None]):
        if coord!=[None,None,None]:
            self.x = coord[2]
            self.y = coord[1]
            self.z = coord[0]
        if self.isTomoLoaded: # test
            slice_xy, slice_zx, slice_zy = self.get_orthoslices(self.vol)
            self.img_xy.setImage(slice_xy, levels=self.levels)
            self.img_zy.setImage(slice_zy, levels=self.levels)
            self.img_zx.setImage(slice_zx, levels=self.levels)
        if self.isLmapLoaded:
            lmap_xy, lmap_zx, lmap_zy = self.get_orthoslices(self.lmap)
            self.img_lmap_xy.setImage(lmap_xy, levels=self.levels_lmap)
            self.img_lmap_zy.setImage(lmap_zy, levels=self.levels_lmap)
            self.img_lmap_zx.setImage(lmap_zx, levels=self.levels_lmap)
        self.lineV_xy.setPos(self.x)
        self.lineH_xy.setPos(self.y)
        self.lineV_zx.setPos(self.x)
        self.lineH_zx.setPos(self.z)
        self.lineV_zy.setPos(self.z)
        self.lineH_zy.setPos(self.y)
        self.set_zoom_centers()

    def mouseClick_secure(self,evt):
        if self.isTomoLoaded == True or self.isLmapLoaded:
            self.mouseClick(evt)
        else:
            display_message_box('Please load a volume first (tomogram or label map)')


    def mouseClick(self,evt):
        pos = evt.scenePos()
        if self.vb_xy.sceneBoundingRect().contains(pos):
            mousePoint = self.vb_xy.mapSceneToView(pos)
            x = int(mousePoint.x())
            y = int(mousePoint.y())
            if x >= 0 and x < self.dim[2] and y >= 0 and y < self.dim[1]:
                self.x = x
                self.y = y
                self.goto_coord()

        if self.vb_zy.sceneBoundingRect().contains(pos):
            mousePoint = self.vb_zy.mapSceneToView(pos)
            y = int(mousePoint.y())
            z = int(mousePoint.x())
            if z >= 0 and z < self.dim[0] and y >= 0 and y < self.dim[1]:
                self.y = y
                self.z = z
                self.goto_coord()

        if self.vb_zx.sceneBoundingRect().contains(pos):
            mousePoint = self.vb_zx.mapSceneToView(pos)
            x = int(mousePoint.x())
            z = int(mousePoint.y())
            if x >= 0 and x < self.dim[2] and z >= 0 and z < self.dim[0]:
                self.x = x
                self.z = z
                self.goto_coord()

        self.label.setText('(x,y,z)=' + '(' + str(self.x) + ',' + str(self.y) + ',' + str(self.z) + ')')

    def set_zoom_centers(self):
        self.vb_xy.set_zoom_center(self.x, self.y)
        self.vb_zy.set_zoom_center(self.z, self.y)
        self.vb_zx.set_zoom_center(self.x, self.z)

    def denoise_slices(self, n): # by averaging neighboring slices
        if n==0: # return noisy slices
            slice_xy, slice_zx, slice_zy = self.get_orthoslices(self.vol)
        else:
            min0 = self.z-n
            max0 = self.z+n
            if min0 < 0          : min0 = 0
            if max0 > self.dim[0]: max0 = self.dim[0]
            slice_xy = self.vol[min0:max0, :, :]
            slice_xy = np.mean(slice_xy, axis=0)
            slice_xy = np.transpose(slice_xy)

            min1 = self.y - n
            max1 = self.y + n
            if min1 < 0          : min1 = 0
            if max1 > self.dim[1]: max1 = self.dim[1]
            slice_zx = self.vol[:, min1:max1, :]
            slice_zx = np.mean(slice_zx, axis=1)
            slice_zx = np.transpose(slice_zx)

            min2 = self.z - n
            max2 = self.z + n
            if min2 < 0          : min2 = 0
            if max2 > self.dim[2]: max2 = self.dim[2]
            slice_zy = self.vol[:, :, min2:max2]
            slice_zy = np.mean(slice_zy, axis=2)

        self.img_xy.setImage(slice_xy, levels=self.levels)
        self.img_zx.setImage(slice_zx, levels=self.levels)
        self.img_zy.setImage(slice_zy, levels=self.levels)


    #def denoise_slices_old(self, sigma): # with bm3d
    #    slice_xy, slice_zx, slice_zy = self.get_orthoslices(self.vol)
    #    slice_xy = cm.denoise2D(slice_xy, sigma)
    #    slice_zx = cm.denoise2D(slice_zx, sigma)
    #    slice_zy = cm.denoise2D(slice_zy, sigma)
    #    self.img_xy.setImage(slice_xy, levels=self.levels)
    #    self.img_zx.setImage(slice_zx, levels=self.levels)
    #    self.img_zy.setImage(slice_zy, levels=self.levels)


# ViewBox has been subclassed to override wheelEvent. In default version, mouse wheel controls zoom towards where the
# mouse points. In present version, the zoom is towards specified (x,y) coordinates, namely the red line cursor.
class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.x_zoom = None
        self.y_zoom = None

    def set_zoom_center(self, x, y):
        self.x_zoom = x
        self.y_zoom = y

    def wheelEvent(self, ev, axis=None):
        mask = np.array(self.state['mouseEnabled'], dtype=np.float)
        if axis is not None and axis >= 0 and axis < len(mask):
            mv = mask[axis]
            mask[:] = 0
            mask[axis] = mv
        s = ((mask * 0.02) + 1) ** (ev.delta() * self.state['wheelScaleFactor'])  # actual scaling factor

        # center = Point(fn.invertQTransform(self.childGroup.transform()).map(ev.pos()))
        center = (self.x_zoom, self.y_zoom) # now zoom towards (x_zoom,y_zoom) instead of where mouse points

        self._resetTarget()
        self.scaleBy(s, center)
        self.sigRangeChangedManually.emit(self.state['mouseEnabled'])
        ev.accept()