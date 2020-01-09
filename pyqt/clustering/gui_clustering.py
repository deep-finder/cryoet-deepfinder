import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

sys.path.append('../../')
import deepfind as df
import utils
import utils_objl as ol

qtcreator_file  = 'gui_clustering.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class ClusteringWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.button_launch.clicked.connect(self.launch_clustering)

    def launch_clustering(self):
        # Get parameters from line edit widgets:
        path_lmap = self.line_path_lmap.text()
        cradius   = int( self.line_cradius.text() )
        csize_thr = int( self.line_csize_thr.text() )
        path_objl = self.line_path_objl.text()

        # Load label map:
        labelmap = utils.read_array(path_lmap)

        # Initialize deepfinder:
        deepfind = df.deepfind(Ncl=13)

        # Launch clustering (result stored in objlist)
        objlist = deepfind.cluster(labelmap, sizeThr=csize_thr, clustRadius=cradius)

        # Save objlist:
        ol.write_xml(objlist, path_objl)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ClusteringWindow()
    window.show()
    sys.exit(app.exec_())