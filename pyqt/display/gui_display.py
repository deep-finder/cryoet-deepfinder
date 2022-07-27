# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import argparse

sys.path.append('../')
from custom_theme import set_custom_theme, display_message_box

from widget_display import DisplayOrthoslicesWidget

sys.path.append('../../')
from cryoet-deepfinder.utils import common as cm

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

        # Signals for communicating with other windows:
        #self.coord_signal = None # for communicating with annotation windows
        self.signal_tomo_loaded = None

    #def connect_coord_signal(self, coord_signal):
    #    self.coord_signal = coord_signal
    def connect_signal_tomo_loaded(self, signal_tomo_loaded):
        self.signal_tomo_loaded = signal_tomo_loaded

    def get_coord(self):
        coord = [self.dwidget.z, self.dwidget.y, self.dwidget.x]
        return coord

    @QtCore.pyqtSlot()
    def on_button_tomo_clicked(self):
        #path_tomo = self.le_path_tomo.text()
        path_tomo =  QtGui.QFileDialog.getOpenFileName(self, 'Open file')
        vol = cm.read_array(path_tomo[0])
        self.set_vol(vol)

    def set_vol(self, vol):
        self.dwidget.set_vol(vol)

        # Set contrast sliders:
        self.slider_contrast_min.setMinimum(self.dataToSliderValue(self.dwidget.vol_min))
        self.slider_contrast_min.setMaximum(self.dataToSliderValue(self.dwidget.vol_mu))
        self.slider_contrast_min.setValue(self.dataToSliderValue(self.dwidget.levels[0]))

        self.slider_contrast_max.setMinimum(self.dataToSliderValue(self.dwidget.vol_mu))
        self.slider_contrast_max.setMaximum(self.dataToSliderValue(self.dwidget.vol_max))
        self.slider_contrast_max.setValue(self.dataToSliderValue(self.dwidget.levels[1]))

        # Automatically propose a value for sigma_noise:
        #sigma_noise = int( 0.7*self.dwidget.vol_sig )
        #self.le_sigma_noise.setText(str(sigma_noise))

        # Inform other windows that tomo is loaded (if they exist):
        if self.signal_tomo_loaded != None:
            self.signal_tomo_loaded.emit()

    @QtCore.pyqtSlot()
    def on_button_lmap_clicked(self):
        #path_lmap = self.le_path_lmap.text()
        path_lmap = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
        lmap = cm.read_array(path_lmap[0])
        self.dwidget.set_lmap(lmap)

    @QtCore.pyqtSlot()
    def on_slider_value_changed_opacity(self):
        if self.dwidget.isLmapLoaded:
            opacity = float(self.slider_opacity.value()) / 100
            self.dwidget.set_lmap_opacity(opacity)
        else:
            display_message_box('Please load a label map first')

    @QtCore.pyqtSlot()
    def on_slider_value_changed_contrast_min(self):
        if self.dwidget.isTomoLoaded:
            levels = self.dwidget.levels
            cmin = self.slider_contrast_min.value()
            cmin = self.sliderToDataValue(float(cmin))
            self.dwidget.set_vol_levels((cmin, levels[1]))
        else:
            display_message_box('Please load a tomogram first')

    @QtCore.pyqtSlot()
    def on_slider_value_changed_contrast_max(self):
        if self.dwidget.isTomoLoaded:
            levels = self.dwidget.levels
            cmax = self.slider_contrast_max.value()
            cmax = self.sliderToDataValue(float(cmax))
            self.dwidget.set_vol_levels((levels[0], cmax))
        else:
            display_message_box('Please load a tomogram first')

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
        if self.dwidget.isTomoLoaded:
            N = int( self.spinb_denoise_param.value() )
            self.dwidget.denoise_slices(N)
        else:
            display_message_box('Please load a tomogram first')

    def place_window(self):
        ag = QtWidgets.QDesktopWidget().availableGeometry()
        width = 3*int(ag.width()/4)
        self.resize(width, ag.height())
        self.move(ag.width() - width, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotate a tomogram.')
    parser.add_argument('-t', action='store', dest='path_tomo', help='path to tomogram')
    parser.add_argument('-l', action='store', dest='path_lmap', help='path to label map')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    set_custom_theme(app)

    win = DisplayWindow()
    win.place_window()
    win.show()

    if args.path_tomo != None:
        tomo = cm.read_array(args.path_tomo)
        win.set_vol(tomo)
        win.button_load_tomo.hide()  # hide load tomo button
    if args.path_lmap != None:
        lmap = cm.read_array(args.path_lmap)
        win.dwidget.set_lmap(lmap)
        win.button_load_lmap.hide()  # hide load lmap button

    sys.exit(app.exec_())