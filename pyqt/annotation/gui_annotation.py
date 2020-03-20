import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

sys.path.append('../')
from custom_theme import set_custom_theme

sys.path.append('../display/')
from gui_display import DisplayWindow

sys.path.append('../../')
from deepfinder.utils import objl as ol
from deepfinder.training import TargetBuilder

import numpy as np

qtcreator_file  = 'gui_annotation.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class AnnotationWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    #coord_signal = QtCore.pyqtSignal(list) # signal for getting coords from display window
    signal_tomo_loaded = QtCore.pyqtSignal()

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # Initialize object list:
        self.objl = []
        self.label_list = [1]

        # Classes section:
        self.table_classes.setColumnCount(2)
        self.table_classes.setRowCount(1)
        self.table_classes.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem(''))
        self.table_classes.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Number of objects'))

        self.table_classes.setItem(0, 0, QtWidgets.QTableWidgetItem('Class 1'))
        self.table_classes.setItem(0, 1, QtWidgets.QTableWidgetItem('0'))

        self.button_classes_add.clicked.connect(self.on_class_add)
        self.button_classes_remove.clicked.connect(self.on_class_remove)

        # Objects section:
        self.table_objects.setColumnCount(3)
        self.table_objects.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Index'))
        self.table_objects.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Class'))
        self.table_objects.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Coordinates'))
        self.table_objects.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

        self.button_objects_add.clicked.connect(self.add_object)
        self.button_objects_remove.clicked.connect(self.on_object_remove)

        self.table_objects.itemSelectionChanged.connect(self.on_object_selected)

        # Set display window:
        self.winDisp = DisplayWindow()
        self.winDisp.button_load_lmap.hide() # hide load lmap button
        #self.winDisp.connect_coord_signal(coord_signal)

        # Connect signals to communicate with display window:
        self.winDisp.connect_signal_tomo_loaded(self.signal_tomo_loaded) # connect to display window
        self.signal_tomo_loaded.connect(self.on_signal_tomo_loaded)      # connect to function

        # Set target builder:
        self.tarBuild = TargetBuilder()
        self.lmap = None
        self.isTomoLoaded = False

    def on_signal_tomo_loaded(self):
        self.isTomoLoaded = True
        tomodim = self.winDisp.dwidget.dim
        self.lmap = np.zeros(tomodim)
        self.winDisp.dwidget.set_lmap(self.lmap)

    def on_class_add(self):
        Nclasses = len(self.label_list)
        self.table_classes.setRowCount(Nclasses + 1)
        new_label = max(self.label_list)+1
        self.label_list.append(new_label)
        self.table_classes.setItem(Nclasses, 0, QtWidgets.QTableWidgetItem('Class ' + str(new_label)))
        self.table_classes.setItem(Nclasses, 1, QtWidgets.QTableWidgetItem('0'))

    def on_class_remove(self):
        row_idx = self.table_classes.currentRow()
        self.table_classes.removeRow(row_idx)
        self.label_list.pop(row_idx)

    # TODO: implement double click on orthoslices for adding obj
    def add_object(self):
        # Get coords:
        coord = self.winDisp.get_coord()
        # Get class label:
        label = self.label_list[ self.table_classes.currentRow() ]
        # Get obj id:
        objid_list = []
        for idx in range(len(self.objl)):
            objid_list.append(self.objl[idx]['obj_id'])
        if len(objid_list)==0: # if 1st object
            new_objid = 1
        else:
            new_objid = max(objid_list)+1
        # Add to objlist:
        self.objl = ol.add_obj(self.objl, label=label, coord=coord, obj_id=new_objid)

        # Add new row to table:
        Nobjects = self.table_objects.rowCount()
        self.table_objects.insertRow(Nobjects)
        self.table_objects.setItem(Nobjects, 0, QtWidgets.QTableWidgetItem(str(new_objid)))
        self.table_objects.setItem(Nobjects, 1, QtWidgets.QTableWidgetItem(str(label)))
        self.table_objects.setItem(Nobjects, 2, QtWidgets.QTableWidgetItem('('+str(coord[2])+','+str(coord[1])+','+str(coord[0])+')'))

        # Count object in table_classes:
        # TODO: problem: when no class is selected, obj count is not displayed
        Nobj_class = len(ol.get_class(self.objl, label))
        self.table_classes.setItem(self.table_classes.currentRow(), 1, QtWidgets.QTableWidgetItem(str(Nobj_class)))

        # Add selected point to display window:
        new_obj = [self.objl[-1]]
        radius_list = [5]*max(self.label_list)
        self.tarBuild.remove_flag = False
        self.lmap = self.tarBuild.generate_with_spheres(new_obj, self.lmap, radius_list)

        self.winDisp.dwidget.update_lmap(self.lmap)
        self.winDisp.dwidget.goto_coord(coord)  # to refresh lmap display

        ol.disp(self.objl)

    # TODO: no remove if no object is selected and no remove if objl is empty
    def on_object_remove(self):
        row_idx = self.table_objects.currentRow()
        objid = int(self.table_objects.item(row_idx, 0).text())
        objl_to_remove = ol.get_obj(self.objl, objid)
        self.objl = ol.remove_obj(self.objl, objid)
        self.table_objects.removeRow(row_idx)
        ol.disp(self.objl)

        x = objl_to_remove[0]['x']
        y = objl_to_remove[0]['y']
        z = objl_to_remove[0]['z']

        # Remove selected point from display window:
        self.tarBuild.remove_flag = True
        radius_list = [5]*max(self.label_list)
        self.lmap = self.tarBuild.generate_with_spheres(objl_to_remove, self.lmap, radius_list)

        self.winDisp.dwidget.update_lmap(self.lmap)
        self.winDisp.dwidget.goto_coord([z, y, x])  # to refresh lmap display



    def on_object_selected(self):
        row_idx = self.table_objects.currentRow()
        objid = int(self.table_objects.item(row_idx,0).text())

        obj = ol.get_obj(self.objl, objid)
        x = obj[0]['x']
        y = obj[0]['y']
        z = obj[0]['z']
        self.winDisp.dwidget.goto_coord([z,y,x])


    def place_windows(self):
        ag = QtWidgets.QDesktopWidget().availableGeometry()
        # Resize and place annotation window:
        winA_w = int(ag.width()/4)
        self.resize(winA_w, ag.height())
        self.move(ag.width() - winA_w, 0)
        # Resize and place display window:
        winD_w = 3*winA_w
        self.winDisp.resize(winD_w, ag.height())
        self.winDisp.move(0,0)


# qtcreator_file  = '../display/gui_display.ui'
# Ui_DispWindow, QtBaseClass = uic.loadUiType(qtcreator_file)
#
# class AnnotationDisplayWindow(QtWidgets.QMainWindow, Ui_DispWindow, DisplayWindow):
#     def __init__(self):
#         QtWidgets.QMainWindow.__init__(self)
#         Ui_DispWindow.__init__(self)
#         self.setupUi(self)
#         DisplayWindow.__init__(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_custom_theme(app)

    win = AnnotationWindow()
    win.place_windows()
    win.winDisp.show()
    win.show()

    sys.exit(app.exec_())