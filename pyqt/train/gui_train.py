import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import os
import threading

sys.path.append('../../')

from deepfinder.training import Train
from deepfinder.utils import core
from deepfinder.utils import objl as ol

qtcreator_file  = 'gui_train.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class TrainingWindow(QtWidgets.QMainWindow, Ui_MainWindow):
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
        Ntomo = int(self.te_path_tomo.document().blockCount())

        path_data = []
        path_target = []
        for idx in range(Ntomo):
            path_data.append(self.te_path_tomo.document().findBlockByLineNumber(idx).text())
            path_target.append(self.te_path_target.document().findBlockByLineNumber(idx).text())

        path_objl_train  = self.le_path_objl_train.text()
        path_objl_valid  = self.le_path_objl_valid.text()
        path_out         = self.le_path_out.text()
        Ncl              = int(self.le_nclass.text())
        psize            = int(self.sb_psize.value())
        bsize            = int( self.le_bsize.text() )
        nepochs          = int( self.le_nepochs.text() )
        steps_per_e      = int( self.le_steps_per_e.text() )
        steps_per_v      = int( self.le_steps_per_v.text() )
        flag_direct_read = self.cb_direct_read.isChecked()
        flag_bootstrap   = self.cb_bootstrap.isChecked()
        rnd_shift        = int( self.le_rnd_shift.text() )

        # Initialize training:
        trainer = Train(Ncl=Ncl)
        trainer.path_out        = path_out
        trainer.dim_in          = psize
        trainer.batch_size      = bsize
        trainer.epochs          = nepochs
        trainer.steps_per_epoch = steps_per_e
        trainer.Nvalid          = steps_per_v
        trainer.flag_direct_read     = flag_direct_read
        trainer.flag_batch_bootstrap = flag_bootstrap
        trainer.Lrnd            = rnd_shift

        trainer.set_observer(core.observer_gui(self.print_signal))

        # Load objlists:
        objl_train = ol.read_xml(path_objl_train)
        objl_valid = ol.read_xml(path_objl_valid)

        # Launch training:
        trainer.launch(path_data, path_target, objl_train, objl_valid)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())