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
import numpy as np

from deepfinder.training import TargetBuilder
from deepfinder.utils import core
from deepfinder.utils import common as cm
from deepfinder.utils import objl as ol


qtcreator_file  = 'gui_target.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class TargetGenerationWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    print_signal = QtCore.pyqtSignal(str) # signal for listening to prints of deepfinder

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.print_signal.connect(self.on_print_signal)
        self.button_launch.clicked.connect(self.on_clicked)

        self.cb_strategy.currentIndexChanged.connect(self.update_label)
        self.cb_initialize.stateChanged.connect(self.show_relevant_widgets)

        # Per default, no target initialization. Therefore, hide following widgets:
        self.lbl_path_vol_initial.hide()
        self.le_path_vol_initial.hide()

    @QtCore.pyqtSlot()
    def show_relevant_widgets(self):
        if self.cb_initialize.isChecked():
            self.lbl_tsize.hide()
            self.lbl_tsize_x.hide()
            self.lbl_tsize_y.hide()
            self.lbl_tsize_z.hide()
            self.le_tsize_x.hide()
            self.le_tsize_y.hide()
            self.le_tsize_z.hide()
            self.lbl_path_vol_initial.show()
            self.le_path_vol_initial.show()
        else:
            self.lbl_tsize.show()
            self.lbl_tsize_x.show()
            self.lbl_tsize_y.show()
            self.lbl_tsize_z.show()
            self.le_tsize_x.show()
            self.le_tsize_y.show()
            self.le_tsize_z.show()
            self.lbl_path_vol_initial.hide()
            self.le_path_vol_initial.hide()

    @QtCore.pyqtSlot()
    def update_label(self):
        if self.cb_strategy.currentText() == 'Shapes':
            self.lbl_strategy_input.setText('Shape mask paths')
        else:
            self.lbl_strategy_input.setText('Sphere radius list')

    @QtCore.pyqtSlot(str)
    def on_print_signal(self, message): # is called when signal is emmited. Signal passes str 'message' to slot
        self.te_terminal_out.append(message)

    @QtCore.pyqtSlot()
    def on_clicked(self):
        threading.Thread(target=self.launch_process, daemon=True).start()

    def launch_process(self):
        # Get parameters from GUI:
        path_objl        = self.le_path_objl.text()
        strategy         = self.cb_strategy.currentText()
        dim_x            = int(self.le_tsize_x.text())
        dim_y            = int(self.le_tsize_y.text())
        dim_z            = int(self.le_tsize_z.text())
        path_initial_vol = self.le_path_vol_initial.text()
        path_target      = self.le_path_target.text()

        Nclass = int(self.te_strategy_input.document().blockCount())
        param_list = [] # for 'shapes' contains mask paths->str, for 'spheres' contains radii->int
        for idx in range(Nclass): # one line corresponds to one class
            param_list.append(self.te_strategy_input.document().findBlockByLineNumber(idx).text())

        # Load objl:
        objl = ol.read_xml(path_objl)

        # Initialize target generation:
        tbuild = TargetBuilder()
        tbuild.set_observer(core.observer_gui(self.print_signal))

        if self.cb_initialize.isChecked():
            tbuild.display('Loading initial volume ...')
            vol_initial = cm.read_array(path_initial_vol)
        else:
            vol_initial = np.zeros((dim_z, dim_y, dim_x))

        if strategy == 'Shapes':
            mask_list = []
            for fname in param_list:  # load masks
                mask = cm.read_array(fname)
                mask_list.append(mask)
            target = tbuild.generate_with_shapes(objl, vol_initial, mask_list)
        else:
            param_list = list(map(int, param_list))  # convert the radius list from str to int
            target = tbuild.generate_with_spheres(objl, vol_initial, param_list)

        tbuild.display('Saving target ...')
        cm.write_array(target, path_target)
        tbuild.display('Done!')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_custom_theme(app)
    window = TargetGenerationWindow()
    window.show()
    sys.exit(app.exec_())