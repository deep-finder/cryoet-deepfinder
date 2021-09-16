# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import argparse

sys.path.append('../')
from custom_theme import set_custom_theme, display_message_box

sys.path.append('../display/')
from gui_display import DisplayWindow

sys.path.append('../../')
from deepfinder.utils import objl as ol
from deepfinder.training import TargetBuilder
from deepfinder.utils import common as cm

import numpy as np

qtcreator_file  = 'gui_annotation.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

# TODO: rename classes
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
        self.path_objl = None

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
        self.table_objects.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Obj ID'))
        self.table_objects.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Class'))
        self.table_objects.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Coordinates'))

        self.table_objects.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

        self.button_objects_add.clicked.connect(self.on_add_object_secure)
        self.button_objects_remove.clicked.connect(self.on_object_remove_secure)

        self.table_objects.itemSelectionChanged.connect(self.on_object_selected)

        # Open&Save buttons:
        self.button_save.clicked.connect(self.on_button_save)
        self.button_open.clicked.connect(self.on_button_open)

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

    def get_selected_rows(self, tableWidget):
        selected_rows = []
        for idx in tableWidget.selectedIndexes():
            selected_rows.append(idx.row())
        selected_rows.sort(reverse=True)
        return selected_rows

    def on_class_add(self):
        Nclasses = len(self.label_list)
        self.table_classes.setRowCount(Nclasses + 1)
        if len(self.label_list)==0:
            new_label = 1
        else:
            new_label = max(self.label_list)+1
        self.label_list.append(new_label)
        self.table_classes.setItem(Nclasses, 0, QtWidgets.QTableWidgetItem('Class ' + str(new_label)))
        self.table_classes.setItem(Nclasses, 1, QtWidgets.QTableWidgetItem('0'))

    def on_class_remove(self):
        if len(self.label_list)>0:
            selected_rows = self.get_selected_rows(self.table_classes)

            # Count number of objects of selected class(es):
            Nobj = 0
            lbl_list_to_remove = []
            for row in selected_rows:
                # Count objects:
                Nobj += int(self.table_classes.item(row, 1).text())
                # Get labels to remove:
                lbl_string = self.table_classes.item(row, 0).text()
                lbl = int(lbl_string[-1]) # 'Class 1' -> 1
                lbl_list_to_remove.append(lbl)

            if Nobj==0: # if class is empty, delete right away
                for row in selected_rows:  # delete
                    self.table_classes.removeRow(row)
                    self.label_list.pop(row)
            else:       # else ask if user is sure
                message = str(Nobj)+' objects will be removed. Proceed?'
                reply = QtGui.QMessageBox.question(self, 'Remove class', message,
                                                   QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                if reply == QtGui.QMessageBox.Yes:
                    # Remove from table_objects and objl:
                    objid_list = []
                    for lbl in lbl_list_to_remove:
                        objl = ol.get_class(self.objl, lbl)
                        for idx in range(len(objl)):
                            objid_list.append(objl[idx]['obj_id'])
                    self.remove_objects(objid_list)

                    # Remove from table_classes:
                    for row in selected_rows:
                        self.table_classes.removeRow(row)
                        self.label_list.pop(row)

                    # Test:
                    print('Remove class!')
                    ol.disp(self.objl)

        else:
            display_message_box('Class list is already empty')

    def on_add_object_secure(self):
        selected_rows = self.get_selected_rows(self.table_classes)

        if self.winDisp.dwidget.isTomoLoaded == False:
            display_message_box('Please load a tomogram first')
        elif len(selected_rows)==0:
            display_message_box('Please select a class before adding an object')
        elif len(selected_rows)>1:
            display_message_box('Only one class at a time must be selected when adding an object')
        else:
            self.on_add_object()


    # TODO: implement double click on orthoslices for adding obj
    def on_add_object(self):
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

        self.add_object(label=label, coord=coord, obj_id=new_objid)

    def add_object(self, label, coord, obj_id):
        # Add to objlist:
        self.objl = ol.add_obj(self.objl, label=label, coord=coord, obj_id=obj_id)

        # Add new row to table:
        Nobjects = self.table_objects.rowCount()
        self.table_objects.insertRow(Nobjects)
        self.table_objects.setItem(Nobjects, 0, QtWidgets.QTableWidgetItem(str(obj_id)))
        self.table_objects.setItem(Nobjects, 1, QtWidgets.QTableWidgetItem(str(label)))
        self.table_objects.setItem(Nobjects, 2, QtWidgets.QTableWidgetItem('('+str(coord[2])+','+str(coord[1])+','+str(coord[0])+')'))

        # Count object in table_classes:
        Nobj_class = len(ol.get_class(self.objl, label))
        self.table_classes.setItem(self.table_classes.currentRow(), 1, QtWidgets.QTableWidgetItem(str(Nobj_class)))

        # Add selected point to display window:
        new_obj = [self.objl[-1]]
        radius_list = [5]*max(self.label_list)
        self.tarBuild.remove_flag = False
        self.lmap = self.tarBuild.generate_with_spheres(new_obj, self.lmap, radius_list)

        self.winDisp.dwidget.update_lmap(self.lmap)
        self.winDisp.dwidget.goto_coord(coord)  # to refresh lmap display

        # Test:
        ol.disp(self.objl)


    def on_object_remove_secure(self):
        if len(self.objl)>0:
            self.on_object_remove()
        else:
            display_message_box('Object list is already empty')

    def on_object_remove(self):
        selected_rows = self.get_selected_rows(self.table_objects)
        objid_list =  []
        for row in selected_rows:
            objid_list.append( int(self.table_objects.item(row, 0).text()) )

        self.remove_objects(objid_list)

        # Update object count in table_classes:
        label = self.label_list[self.table_classes.currentRow()]
        Nobj_class = len(ol.get_class(self.objl, label))
        self.table_classes.setItem(self.table_classes.currentRow(), 1, QtWidgets.QTableWidgetItem(str(Nobj_class)))

        ol.disp(self.objl)

    def remove_objects(self, objid_list):
        # Remove objects from objl:
        objl_to_remove = ol.get_obj(self.objl, objid_list)
        self.objl = ol.remove_obj(self.objl, objid_list)

        # Get table row idx of objects to remove:
        Nobj = self.table_objects.rowCount()
        rows_to_remove = []
        for row in range(Nobj):
            objid = int(self.table_objects.item(row, 0).text())
            for idx in range(len(objl_to_remove)):
                if objid == objl_to_remove[idx]['obj_id']:
                    rows_to_remove.append(row)
        rows_to_remove.sort(reverse=True)

        # Remove objects from table:
        for row in rows_to_remove:
            self.table_objects.removeRow(row)

        # Remove objects from display window:
        self.tarBuild.remove_flag = True
        radius_list = [5] * max(self.label_list)
        self.lmap = self.tarBuild.generate_with_spheres(objl_to_remove, self.lmap, radius_list)

        self.winDisp.dwidget.update_lmap(self.lmap)
        self.winDisp.dwidget.goto_coord()  # to refresh lmap display

    def get_checked_list(self, tableWidget, col):
        Nrows = tableWidget.rowCount()
        checked_list = []
        for row in range(Nrows):
            if tableWidget.cellWidget(row,col).isChecked():
                checked_list.append(row)
        return checked_list

    def on_object_selected(self):
        selected_rows = self.get_selected_rows(self.table_objects)
        if len(selected_rows)==1:
            row_idx = selected_rows[0]
            objid = int(self.table_objects.item(row_idx,0).text())
            obj = ol.get_obj(self.objl, [objid])
            if len(obj)==1:
                x = obj[0]['x']
                y = obj[0]['y']
                z = obj[0]['z']
                self.winDisp.dwidget.goto_coord([z,y,x])

    def on_button_save(self):
        if len(self.objl)==0:
            display_message_box('The object list is empty')
        else:
            if self.path_objl == None: # if path not specified by -o option, then open dialog window
                filename = QtGui.QFileDialog.getSaveFileName(self, 'Save object list')
                filename = filename[0]
            else:
                filename = self.path_objl
            s = os.path.splitext(filename)
            filename = s[0]+'.xml' # force extension to be xml
            ol.write_xml(self.objl, filename)

            message = 'Object list saved! Quit?'
            reply = QtGui.QMessageBox.question(self, 'Quit?', message,
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                sys.exit()


    def on_button_open(self):
        if self.winDisp.dwidget.isTomoLoaded == False:
            display_message_box('Please load a tomogram first')
        else:
            path_objl = ('', '')
            if len(self.objl)!=0:
                message = 'This will overwrite current object list. Proceed?'
                reply = QtGui.QMessageBox.question(self, 'Remove class', message,
                                                   QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                if reply == QtGui.QMessageBox.Yes:
                    path_objl = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
            else:
                path_objl = QtGui.QFileDialog.getOpenFileName(self, 'Open file')

            if path_objl[0]!='':
                self.load_objl(path_objl[0])




    def load_objl(self, path_objl):
        objl = ol.read_xml(path_objl)
        self.label_list = ol.get_labels(objl)

        for idx in range(len(objl)):
            x = int( objl[idx]['x'] )
            y = int( objl[idx]['y'] )
            z = int( objl[idx]['z'] )
            label = int( objl[idx]['label'] )
            obj_id = int( objl[idx]['obj_id'] )
            print('label: '+str(label))
            self.add_object(label, [z, y, x], obj_id)



    def place_windows(self):
        ag = QtWidgets.QDesktopWidget().availableGeometry()
        # Resize and place annotation window:
        winA_w = int(ag.width()/4)
        self.resize(winA_w, ag.height())
        self.move(0, 0)
        # Resize and place display window:
        winD_w = 3*winA_w
        self.winDisp.resize(winD_w, ag.height())
        self.winDisp.move(ag.width() - winD_w,0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotate a tomogram.')
    parser.add_argument('-t', action='store', dest='path_tomo', help = 'path to tomogram')
    parser.add_argument('-o', action='store', dest='path_objl', help = 'output path for object list')
    parser.add_argument('-scipion', action='store_true', help='option for launching in scipion (hides some buttons)')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    set_custom_theme(app)

    win = AnnotationWindow()
    win.place_windows()
    win.winDisp.show()
    win.show()

    if args.path_tomo != None:
        tomo = cm.read_array(args.path_tomo)
        win.winDisp.set_vol(tomo)
        win.winDisp.button_load_tomo.hide()  # hide load tomo button
    if args.path_objl != None:
        win.path_objl = args.path_objl
    if args.scipion:
        win.button_open.hide()

    sys.exit(app.exec_())