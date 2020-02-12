import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

sys.path.append('../')
from custom_theme import set_custom_theme

import os
import threading

sys.path.append('../../')
from deepfinder.inference import Segment
from deepfinder.utils import core
from deepfinder.utils import common as cm
from deepfinder.utils import smap as sm

qtcreator_file  = 'gui_segment.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class SegmentationWindow(QtWidgets.QMainWindow, Ui_MainWindow):
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
        threading.Thread(target=self.launch_process, daemon=True).start()

    def launch_process(self):
        # Get parameters from line edit widgets:
        Ncl          = int( self.le_nclass.text() )
        psize        = int( self.sb_psize.value() )
        path_weights = self.le_path_weights.text()
        path_data    = self.le_path_tomo.text()
        path_lmap    = self.le_path_lmap.text()


        # Load data:
        data = cm.read_array(path_data)

        # Initialize segmentation:
        seg = Segment(Ncl=Ncl, path_weights=path_weights, patch_size=psize)
        seg.set_observer(core.observer_gui(self.print_signal))

        # Segment data:
        scoremaps = seg.launch(data)

        seg.display('Saving labelmap ...')
        # Get labelmap from scoremaps and save:
        labelmap = sm.to_labelmap(scoremaps)
        cm.write_array(labelmap, path_lmap)

        # Get binned labelmap and save:
        if self.cb_bin.isChecked():
            s = os.path.splitext(path_lmap)
            scoremapsB = sm.bin(scoremaps)
            labelmapB = sm.to_labelmap(scoremapsB)
            cm.write_array(labelmapB, s[0]+'_binned'+s[1])

        seg.display('Finished !')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_custom_theme(app)
    window = SegmentationWindow()
    window.show()
    sys.exit(app.exec_())