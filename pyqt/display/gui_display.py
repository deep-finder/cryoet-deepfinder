import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic


from widget_display import DisplayOrthoslicesWidget
from deepfinder.utils import common as cm


qtcreator_file  = 'gui_display.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class DisplayWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.dwidget = DisplayOrthoslicesWidget()
        self.layout.addWidget(self.dwidget)

        self.button_load.clicked.connect(self.on_button_clicked)


    @QtCore.pyqtSlot()
    def on_button_clicked(self):
        path_data = self.le_path.text()
        vol = cm.read_array(path_data)
        self.dwidget.set_vol(vol)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DisplayWindow()
    window.show()
    sys.exit(app.exec_())