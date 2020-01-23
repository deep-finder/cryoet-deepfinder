import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import threading

sys.path.append('../../')
import deepfind as df
import core_utils
import utils
import utils_objl as ol


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
        path_initial_vol = self.le_initial_vol.text()
        path_target      = self.le_path_target.text

        # Initialize target generation:
        tbuild = df.TargetBuilder()
        tbuild.set_observer(core_utils.observer_gui(self.print_signal))

        if path_initial_vol=='(optional)' or path_initial_vol = None:
            vol_initial = np.zeros(tomodim)
        else:
            tbuild.display('Loading initial volume ...')
            vol_initial = utils.read_array(path_initial_vol)

        if strategy == 'Shapes':
            target = tbuild.generate_with_shapes(objl, np.zeros(tomodim), ref_list)
        else:
            target = tbuild.generate_with_spheres(objl, np.zeros(tomodim), radius_list)


        utils.write_array(target, 'out/target_tomo42.mrc')
        print('pouet')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TargetGenerationWindow()
    window.show()
    sys.exit(app.exec_())