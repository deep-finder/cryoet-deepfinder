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

import threading

sys.path.append('../../')

from deepfinder.inference import Cluster
from deepfinder.utils import core
from deepfinder.utils import common as cm
from deepfinder.utils import objl as ol


qtcreator_file  = 'gui_clustering.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class ClusteringWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    print_signal = QtCore.pyqtSignal(str) # signal for listening to prints of deepfinder

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.print_signal.connect(self.on_print_signal)
        self.button_launch.clicked.connect(self.on_clicked)

    @QtCore.pyqtSlot(str)
    def on_print_signal(self, message): # is called when signal is emmited. Signal passes str 'message' to slot
        self.te_terminal_out.append(message)

    @QtCore.pyqtSlot()
    def on_clicked(self):
        threading.Thread(target=self.launch_process, daemon=True).start()

    def launch_process(self):
        # Get parameters from line edit widgets:
        path_lmap = self.le_path_lmap.text()
        cradius   = int( self.le_cradius.text() )
        csize_thr = int( self.le_csize_thr.text() )
        path_objl = self.le_path_objl.text()

        # Initialize deepfinder:
        clust = Cluster(clustRadius=cradius)
        clust.sizeThr = csize_thr
        clust.set_observer(core.observer_gui(self.print_signal))

        # Load label map:
        clust.display('Loading label map ...')
        labelmap = cm.read_array(path_lmap)

        # Launch clustering (result stored in objlist)
        objlist = clust.launch(labelmap)

        # Save objlist:
        ol.write(objlist, path_objl)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_custom_theme(app)
    window = ClusteringWindow()
    window.show()
    sys.exit(app.exec_())