# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

sys.path.append('../')
from custom_theme import set_custom_theme

sys.path.append('../display/')
from gui_display import DisplayWindow

import os
import threading

sys.path.append('../../')
from deepfinder.inference import Segment
from deepfinder.utils import core
from deepfinder.utils import common as cm
from deepfinder.utils import smap as sm

qtcreator_file  = 'gui_segment.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

# TODO: option for choosing segm strategy: one pass or in patch
class SegmentationWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    print_signal = QtCore.pyqtSignal(str) # signal for listening to prints of deepfinder

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.print_signal.connect(self.on_print_signal)
        self.button_launch.clicked.connect(self.on_clicked)

        # Set display window:
        self.winDisp = DisplayWindow()
        self.winDisp.button_load_tomo.hide()  # hide load tomo button
        self.winDisp.button_load_lmap.hide()  # hide load lmap button

        self.data = None # for some reason, winDisp cannor be called from thread, so data and lmap are stored here for access
        self.labelmap = None

    @QtCore.pyqtSlot(str)
    def on_print_signal(self, message): # is called when signal is emmited. Signal passes str 'message' to slot
        self.te_terminal_out.append(message)

        if message == 'Finished !':
            # Display result:
            self.winDisp.show()
            self.place_window_display()
            self.winDisp.dwidget.set_vol(self.data)
            self.winDisp.dwidget.set_lmap(self.labelmap)

    def on_clicked(self):
        threading.Thread(target=self.launch_process, daemon=True).start()

    def launch_process(self):
        # Get parameters from line edit widgets:
        Ncl          = int( self.le_nclass.text() )
        psize        = int( self.sb_psize.value() )
        path_weights = self.le_path_weights.text()
        path_data    = self.le_path_tomo.text()
        path_lmap    = self.le_path_lmap.text()


        # Load data:
        self.data = cm.read_array(path_data)

        # Initialize segmentation:
        seg = Segment(Ncl=Ncl, path_weights=path_weights, patch_size=psize)
        seg.set_observer(core.observer_gui(self.print_signal))

        # Segment data:
        scoremaps = seg.launch(self.data)

        seg.display('Saving labelmap ...')
        # Get labelmap from scoremaps and save:
        self.labelmap = sm.to_labelmap(scoremaps)
        cm.write_array(self.labelmap, path_lmap)

        # Get binned labelmap and save:
        if self.cb_bin.isChecked():
            s = os.path.splitext(path_lmap)
            scoremapsB = sm.bin(scoremaps)
            labelmapB = sm.to_labelmap(scoremapsB)
            cm.write_array(labelmapB, s[0]+'_binned'+s[1])

        seg.display('Finished !')



    def place_window_segment(self):
        ag = QtWidgets.QDesktopWidget().availableGeometry()
        win_w = int(ag.width() / 4)
        win_h = 2 * int(ag.height() / 3)
        self.resize(win_w, win_h)
        self.move(0, 0)

    def place_window_display(self):
        ag = QtWidgets.QDesktopWidget().availableGeometry()
        win_w = int(3 * ag.width() / 4)
        win_h = 2 * int(ag.height() / 3)
        self.winDisp.resize(win_w, win_h)
        self.winDisp.move(int(ag.width() / 4),0)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_custom_theme(app)
    win = SegmentationWindow()
    win.show()
    win.place_window_segment()
    sys.exit(app.exec_())