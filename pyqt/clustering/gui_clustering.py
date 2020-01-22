import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import threading

sys.path.append('../../')
import deepfind as df
import core_utils
import utils
import utils_objl as ol

from PyQt5.QtCore import QThread

# class Worker(QThread):
#     def __init__(self, clust, path_lmap, path_objl):
#         QThread.__init__(self)
#         self.clust  = clust
#         self.path_lmap = path_lmap
#         self.path_objl = path_objl
#     def start(self):
#         # Load label map:
#         labelmap = utils.read_array(self.path_lmap)
#
#         # Launch clustering (result stored in objlist)
#         objlist = self.clust.launch(labelmap)
#
#         # Save objlist:
#         ol.write_xml(objlist, self.path_objl)



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
        threading.Thread(target=self.launch_clustering, daemon=True).start()

        # # Get parameters from line edit widgets:
        # path_lmap = self.line_path_lmap.text()
        # cradius = int(self.line_cradius.text())
        # csize_thr = int(self.line_csize_thr.text())
        # path_objl = self.line_path_objl.text()
        #
        # # # Initialize deepfinder:
        # clust = df.cluster(clustRadius=cradius)
        # clust.sizeThr = csize_thr
        # clust.set_observer(core_utils.observer_gui(self.print_signal))
        #
        # # Create and launch thread:
        # w = Worker(clust, path_lmap, path_objl)
        # w.start()
        # print(str(w.currentThreadId()))

        # dummy = df.dummy()
        # dummy.set_observer(core_utils.observer_gui(self.print_signal))
        # w = DummyWorker(dummy)
        # w.start()

    def launch_clustering(self):
        # Get parameters from line edit widgets:
        path_lmap = self.le_path_lmap.text()
        cradius   = int( self.le_cradius.text() )
        csize_thr = int( self.le_csize_thr.text() )
        path_objl = self.le_path_objl.text()

        # Load label map:
        labelmap = utils.read_array(path_lmap)

        # Initialize deepfinder:
        clust = df.Cluster(clustRadius=cradius)
        clust.sizeThr = csize_thr
        clust.set_observer(core_utils.observer_gui(self.print_signal))

        # Launch clustering (result stored in objlist)
        objlist = clust.launch(labelmap)

        # Display result:
        self.display_result(clust, objlist)

        # Save objlist:
        ol.write_xml(objlist, path_objl)

    # Displays end result
    # INPUTS:
    #   clust: Cluster object, is needed to access clust.display() method
    #   objlist: input objl
    def display_result(self, clust, objlist):
        clust.display('A total of ' + str(len(objlist)) + ' objects have been found.')
        lbl_list = ol.get_labels(objlist)
        for lbl in lbl_list:
            objl_class = ol.get_class(objlist, lbl)
            clust.display('Class ' + str(lbl) + ': ' + str(len(objl_class)) + ' objects')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ClusteringWindow()
    window.show()
    sys.exit(app.exec_())