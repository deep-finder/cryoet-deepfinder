import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

sys.path.append('../')
from custom_theme import set_custom_theme

from widget_display import DisplayOrthoslicesWidget
from deepfinder.utils import common as cm

import gui_display_interface

#qtcreator_file  = 'gui_display.ui'
#Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

#class DisplayWindow(QtWidgets.QMainWindow, Ui_MainWindow):
class DisplayWindow(QtWidgets.QMainWindow, gui_display_interface.Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        gui_display_interface.Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.dwidget = DisplayOrthoslicesWidget()
        self.layout.addWidget(self.dwidget)

        #self.histogram_widget.setImageItem(self.dwidget.img_xy)
        #self.histogram_widget.setImageItem(self.dwidget.img_zx)
        #self.histogram_widget.setImageItem(self.dwidget.img_zy)

        self.button_load_tomo.clicked.connect(self.on_button_tomo_clicked)
        self.button_load_lmap.clicked.connect(self.on_button_lmap_clicked)
        self.slider_opacity.valueChanged.connect(self.on_slider_value_changed_opacity)
        self.slider_contrast_min.valueChanged.connect(self.on_slider_value_changed_contrast_min)
        self.slider_contrast_max.valueChanged.connect(self.on_slider_value_changed_contrast_max)

        self.button_denoise.clicked.connect(self.on_button_denoised)

        #self.coord_signal = None # for communicating with annotation windows

    #def connect_coord_signal(self, coord_signal):
    #    self.coord_signal = coord_signal
    def get_coord(self):
        coord = [self.dwidget.z, self.dwidget.y, self.dwidget.x]
        return coord

    @QtCore.pyqtSlot()
    def on_button_tomo_clicked(self):
        path_tomo = self.le_path_tomo.text()
        vol = cm.read_array(path_tomo)
        self.dwidget.set_vol(vol)

        # Set contrast sliders:
        self.slider_contrast_min.setMinimum(self.dataToSliderValue(self.dwidget.vol_min))
        self.slider_contrast_min.setMaximum(self.dataToSliderValue(self.dwidget.vol_mu))
        self.slider_contrast_min.setValue(self.dataToSliderValue(self.dwidget.levels[0]))

        self.slider_contrast_max.setMinimum(self.dataToSliderValue(self.dwidget.vol_mu))
        self.slider_contrast_max.setMaximum(self.dataToSliderValue(self.dwidget.vol_max))
        self.slider_contrast_max.setValue(self.dataToSliderValue(self.dwidget.levels[1]))

        # Automatically propose a value for sigma_noise:
        sigma_noise = int( 0.7*self.dwidget.vol_sig )
        self.le_sigma_noise.setText(str(sigma_noise))

    @QtCore.pyqtSlot()
    def on_button_lmap_clicked(self):
        path_lmap = self.le_path_lmap.text()
        lmap = cm.read_array(path_lmap)
        self.dwidget.set_lmap(lmap)

    @QtCore.pyqtSlot()
    def on_slider_value_changed_opacity(self):
        opacity = float(self.slider_opacity.value()) / 100
        self.dwidget.set_lmap_opacity(opacity)

    @QtCore.pyqtSlot()
    def on_slider_value_changed_contrast_min(self):
        levels = self.dwidget.levels
        cmin = self.slider_contrast_min.value()
        #self.slider_contrast_max.setMinimum(cmin)
        cmin = self.sliderToDataValue(float(cmin))
        self.dwidget.set_vol_levels((cmin, levels[1]))

    @QtCore.pyqtSlot()
    def on_slider_value_changed_contrast_max(self):
        levels = self.dwidget.levels
        cmax = self.slider_contrast_max.value()
        #self.slider_contrast_min.setMaximum(cmax)
        cmax = self.sliderToDataValue(float(cmax))
        self.dwidget.set_vol_levels((levels[0], cmax))

    def dataToSliderValue(self, val):
        return 100*(val-self.dwidget.vol_min)/(self.dwidget.vol_max-self.dwidget.vol_min)
    def sliderToDataValue(self, val):
        return val*(self.dwidget.vol_max-self.dwidget.vol_min)/100 + self.dwidget.vol_min

    @QtCore.pyqtSlot()
    def on_zoom_slider_value_changed(self):
        scale = float(self.slider_zoom.value())/10
        self.dwidget.zoom_slices((scale,scale))

    @QtCore.pyqtSlot()
    def on_button_denoised(self):
        sigma_noise = float( self.le_sigma_noise.text() )
        self.dwidget.denoise_slices(sigma_noise)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_custom_theme(app)

    window = DisplayWindow()
    window.show()
    sys.exit(app.exec_())