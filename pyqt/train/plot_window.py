# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

sys.path.append('../')

import numpy as np
import pyqtgraph as pg
from deepfinder.utils import core

# Note: for multiple axe plots, check: pyqtgraph/examples/MultiplePlotAxes.py
class TrainMetricsPlotWindow:
    def __init__(self):
        self.win = pg.GraphicsWindow()  # Automatically generates grids with multiple items
        self.p_loss = self.win.addPlot(row=0, col=0)
        self.p_acc  = self.win.addPlot(row=1, col=0)
        self.p_f1   = self.win.addPlot(row=0, col=1)
        self.p_pre = self.win.addPlot(row=1, col=1)
        self.p_rec  = self.win.addPlot(row=2, col=1)

        # Set Y axis label:
        self.p_loss.setLabel(axis='left', text='Loss')
        self.p_acc.setLabel(axis='left', text='Accuracy')
        self.p_f1.setLabel(axis='left', text='F1-score')
        self.p_pre.setLabel(axis='left', text='Precision')
        self.p_rec.setLabel(axis='left', text='Recall')

        # Set X axis label:
        self.p_loss.setLabel(axis='bottom', text='epochs')
        self.p_acc.setLabel(axis='bottom', text='epochs')
        self.p_f1.setLabel(axis='bottom', text='epochs')
        self.p_pre.setLabel(axis='bottom', text='epochs')
        self.p_rec.setLabel(axis='bottom', text='epochs')

        # Set grid:
        self.p_loss.showGrid(x=True, y=True)
        self.p_acc.showGrid(x=True, y=True)
        self.p_f1.showGrid(x=True, y=True)
        self.p_pre.showGrid(x=True, y=True)
        self.p_rec.showGrid(x=True, y=True)


    def update_plots(self, filename):
        history = core.read_history(filename)

        Ncl = history['val_f1'].shape[2]
        epochs = len(history['val_loss'])
        steps_per_valid = len(history['val_loss'][0])

        # Average over valid_steps:
        val_loss = np.mean(history['val_loss'], axis=1)
        val_acc = np.mean(history['val_acc'], axis=1)
        val_f1 = np.mean(history['val_f1'], axis=1)
        val_pre = np.mean(history['val_precision'], axis=1)
        val_rec = np.mean(history['val_recall'], axis=1)

        loss = []
        acc = []
        for e in range(epochs):
            loss.append(np.mean(history['loss'][e, -steps_per_valid:]))
            acc.append(np.mean(history['acc'][e, -steps_per_valid:]))

        # Plot loss and accuracy:
        self.p_loss.plot(loss, pen=pg.intColor(0), name='Train loss')
        self.p_loss.plot(val_loss, pen=pg.intColor(1), name='Valid loss')

        self.p_acc.plot(acc, pen=pg.intColor(0), name='Train loss')
        self.p_acc.plot(val_acc, pen=pg.intColor(1), name='Valid loss')

        # Plot f1, precision and recall:
        for cl in range(Ncl):
            self.p_f1.plot(val_f1[:,cl], pen=pg.intColor(2*cl+1), name='Class '+str(cl))
            self.p_pre.plot(val_pre[:,cl], pen=pg.intColor(2*cl+1), name='Class '+str(cl))
            self.p_rec.plot(val_rec[:,cl], pen=pg.intColor(2*cl+1), name='Class '+str(cl))

        self.p_loss.addLegend(offset=(-20, 1))
        self.p_f1.addLegend(offset=(-20, -0))









#qtcreator_file  = 'gui_display.ui'
#Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

#class DisplayWindow(QtWidgets.QMainWindow, Ui_MainWindow):
# class DisplayWindow(QtWidgets.QMainWindow, gui_display_interface.Ui_MainWindow):
#     def __init__(self):
#         QtWidgets.QMainWindow.__init__(self)
#         gui_display_interface.Ui_MainWindow.__init__(self)
#         self.setupUi(self)
#
#         self.dwidget = TrainMetricsPlotWidget()
#         self.layout.addWidget(self.dwidget)


# class TrainMetricsPlotWidget(QWidget):
#     def __init__(self):
#         QWidget.__init__(self)
#
#         self.gl = pg.GraphicsLayoutWidget()
#
#         # This is necessary to attach the GraphicsLayoutWidget to TrainMetricsPlotWidget:
#         self.layout = QVBoxLayout()
#         QWidget.setLayout(self, self.layout)
#         self.layout.addWidget(self.gl)
#
#         self.vb_xy = CustomViewBox(invertY=True)
#         self.gl.addItem(self.vb_xy, row=0, col=0)