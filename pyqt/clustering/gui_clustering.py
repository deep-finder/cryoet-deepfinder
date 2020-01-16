import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import threading

sys.path.append('../../')
import deepfind as df
import core_utils
import utils
import utils_objl as ol

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

    def on_clicked(self):
        threading.Thread(target=self.launch_clustering(), daemon=True).start()
        # dont forget '()' after self.launch_clustering else File ".../tensorflow_backend.py": AttributeError: '_thread._local' object has no attribute 'value'

    def launch_clustering(self):
        # Get parameters from line edit widgets:
        path_lmap = self.line_path_lmap.text()
        cradius   = int( self.line_cradius.text() )
        csize_thr = int( self.line_csize_thr.text() )
        path_objl = self.line_path_objl.text()

        # Load label map:
        labelmap = utils.read_array(path_lmap)

        # Initialize deepfinder:
        clust = df.cluster(clustRadius=cradius)
        clust.sizeThr = csize_thr
        clust.set_observer(core_utils.observer_gui(self.print_signal))

        # Launch clustering (result stored in objlist)
        objlist = clust.launch(labelmap)

        # Save objlist:
        ol.write_xml(objlist, path_objl)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ClusteringWindow()
    window.show()
    sys.exit(app.exec_())