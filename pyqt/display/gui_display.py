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

        self.button_load_tomo.clicked.connect(self.on_button_tomo_clicked)
        self.button_load_lmap.clicked.connect(self.on_button_lmap_clicked)
        self.slider_opacity.valueChanged.connect(self.on_slider_value_changed)

    @QtCore.pyqtSlot()
    def on_button_tomo_clicked(self):
        path_tomo = self.le_path_tomo.text()
        vol = cm.read_array(path_tomo)
        self.dwidget.set_vol(vol)

    @QtCore.pyqtSlot()
    def on_button_lmap_clicked(self):
        path_lmap = self.le_path_lmap.text()
        lmap = cm.read_array(path_lmap)
        self.dwidget.set_lmap(lmap)

    @QtCore.pyqtSlot()
    def on_slider_value_changed(self):
        opacity = float(self.slider_opacity.value()) / 100
        self.dwidget.set_lmap_opacity(opacity)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DisplayWindow()
    window.show()
    sys.exit(app.exec_())