from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from deepfinder.utils import common as cm

class DisplayOrthoslicesWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.gl = pg.GraphicsLayoutWidget()

        # This is necessary to attach the GraphicsLayoutWidget to this DisplayOrthoslicesWidget:
        self.layout = QVBoxLayout()
        QWidget.setLayout(self, self.layout)
        self.layout.addWidget(self.gl)

        # Orthoslices:
        self.vb_xy = self.gl.addViewBox(row=1, col=1, invertY=True)
        self.vb_zy = self.gl.addViewBox(row=1, col=2, invertY=True)
        self.vb_zx = self.gl.addViewBox(row=2, col=1, invertY=True)

        self.img_xy = pg.ImageItem()
        self.img_zy = pg.ImageItem()
        self.img_zx = pg.ImageItem()

        self.vb_xy.addItem(self.img_xy)
        self.vb_zy.addItem(self.img_zy)
        self.vb_zx.addItem(self.img_zx)

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

        # Add orthoslice labels:
        self.gl.addLabel('X', col=1, row=0)
        self.gl.addLabel('Z', col=2, row=0)
        self.gl.addLabel('Y', col=0, row=1)
        self.gl.addLabel('Z', col=0, row=2)

        # Label displaying coordinates:
        self.label = pg.LabelItem()
        self.gl.addItem(self.label, row=2, col=2)

        # Relative to displayed data, needs to be initialized by set_vol()
        self.vol = None
        self.dim = (None, None, None)
        self.x = None
        self.y = None
        self.z = None

        # Connect click signal to dedicated function:
        self.gl.scene().sigMouseClicked.connect(self.mouseClick)

    def set_vol(self, vol):
        #self.vol = cm.read_array(filename)
        self.vol = vol
        self.dim = self.vol.shape
        self.x = np.int(np.round(self.dim[2] / 2))
        self.y = np.int(np.round(self.dim[1] / 2))
        self.z = np.int(np.round(self.dim[0] / 2))

        slice_xy = np.transpose(self.vol[self.z, :, :])
        slice_zx = np.transpose(self.vol[:, self.y, :])
        slice_zy = self.vol[:, :, self.x]

        self.img_xy.setImage(slice_xy)
        self.img_zx.setImage(slice_zx)
        self.img_zy.setImage(slice_zy)

        self.lineV_xy.setPos(self.x)
        self.lineH_xy.setPos(self.y)

        self.lineV_zy.setPos(self.z)
        self.lineH_zy.setPos(self.y)

        self.lineV_zx.setPos(self.x)
        self.lineH_zx.setPos(self.z)

        self.vb_xy.setLimits(xMin=0, xMax=self.dim[2], yMin=0, yMax=self.dim[1])
        self.vb_zx.setLimits(xMin=0, xMax=self.dim[2], yMin=0, yMax=self.dim[0])
        self.vb_zy.setLimits(xMin=0, xMax=self.dim[0], yMin=0, yMax=self.dim[1])

    def mouseClick(self,evt):
        pos = evt.scenePos()
        if self.vb_xy.sceneBoundingRect().contains(pos):
            mousePoint = self.vb_xy.mapSceneToView(pos)
            x = int(mousePoint.x())
            y = int(mousePoint.y())
            if x >= 0 and x < self.dim[2] and y >= 0 and y < self.dim[1]:
                self.x = x
                self.y = y
                slice_zx = np.transpose(self.vol[:, y, :])
                slice_zy = self.vol[:, :, x]
                self.img_zy.setImage(slice_zy)
                self.img_zx.setImage(slice_zx)
                self.lineV_xy.setPos(x)
                self.lineH_xy.setPos(y)
                self.lineV_zx.setPos(x)
                self.lineH_zy.setPos(y)

        if self.vb_zy.sceneBoundingRect().contains(pos):
            mousePoint = self.vb_zy.mapSceneToView(pos)
            y = int(mousePoint.y())
            z = int(mousePoint.x())
            if z >= 0 and z < self.dim[0] and y >= 0 and y < self.dim[1]:
                self.y = y
                self.z = z
                slice_xy = np.transpose(self.vol[z, :, :])
                slice_zx = np.transpose(self.vol[:, y, :])
                self.img_xy.setImage(slice_xy)
                self.img_zx.setImage(slice_zx)
                self.lineV_zy.setPos(z)
                self.lineH_zy.setPos(y)
                self.lineH_xy.setPos(y)
                self.lineH_zx.setPos(z)

        if self.vb_zx.sceneBoundingRect().contains(pos):
            mousePoint = self.vb_zx.mapSceneToView(pos)
            x = int(mousePoint.x())
            z = int(mousePoint.y())
            if x >= 0 and x < self.dim[2] and z >= 0 and z < self.dim[0]:
                self.x = x
                self.z = z
                slice_xy = np.transpose(self.vol[z, :, :])
                slice_zy = self.vol[:, :, x]
                self.img_xy.setImage(slice_xy)
                self.img_zy.setImage(slice_zy)
                self.lineV_zx.setPos(x)
                self.lineH_zx.setPos(z)
                self.lineV_xy.setPos(x)
                self.lineV_zy.setPos(z)

        self.label.setText('(x,y,z)=' + '(' + str(self.x) + ',' + str(self.y) + ',' + str(self.z) + ')')
