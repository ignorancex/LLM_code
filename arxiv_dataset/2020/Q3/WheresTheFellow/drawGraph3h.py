import sys
import json
from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtCore import QRect
import subprocess
from graph_generator import HumanGraph
import math

import ui_drawGraph


parts = ['sb', 'b', 'n', 'le', 'ley', 'ls', 'lel', 'lw', 'lh', 'lk', 'la', 're', 'rey', 'rs', 'rel', 'rw', 'rh', 'rk', 'ra']
#
# TODO list:
#
# - Avoid creating a new instance of MyView every time a new scenario is shown. For this we would only need to
#   modify MyView so that it updates the scenario instead of creating new ones.
#
#
#



class MainClass(QtWidgets.QWidget):
    def __init__(self, scenarios, start):
        super().__init__()
        self.scenarios = scenarios
        self.ui = ui_drawGraph.Ui_CalibrationWidget()
        self.ui.setupUi(self)
        self.next_index = start
        self.view = None
        self.show()
        self.installEventFilter(self)
        self.load_next()
        self.ui.tableWidget.setRowCount(self.view.graph.features.shape[1]+1)
        self.ui.tableWidget.setColumnCount(1)
        self.ui.tableWidget.setColumnWidth(0, 200)
        self.ui.tableWidget.show()

        # Initialize table
        self.ui.tableWidget.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('value'))
        self.ui.tableWidget.horizontalHeader().hide()
        self.ui.tableWidget.setVerticalHeaderItem(0, QtWidgets.QTableWidgetItem('type'))
        self.ui.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem('0'))
        features_aux = self.view.graph.get_all_features()
        for idx, feature in enumerate(features_aux, 1):
            self.ui.tableWidget.setVerticalHeaderItem(idx, QtWidgets.QTableWidgetItem(feature))
            self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))

    def load_next(self):
        if self.view:
            self.view.close()
            del self.view
        self.view = MyView(self.scenarios[self.next_index], self.ui.tableWidget)
        self.next_index += 1
        self.view.setParent(self.ui.widget)
        self.view.show()
        self.ui.widget.setFixedSize(self.view.width(), self.view.height())
        self.show()

        # Initialize table with zeros
        for idx in range(self.view.graph.features.shape[1]+2):
            self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))

    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Escape:
                sys.exit(0)
            else:
                if key == QtCore.Qt.Key_Return:
                    self.load_next()
                elif key == QtCore.Qt.Key_Enter:
                    self.close()
                return True
        return False


class MyView(QtWidgets.QGraphicsView):
    def __init__(self, scenario, table):
        super().__init__()
        self.table = table
        self.setFixedSize(1002, 1002)
        self.create_scene(scenario)
        self.installEventFilter(self)

    def create_scene(self, human):

        self.graph = HumanGraph(human, '1', mode='run', debug=True)
        self.all_types = dict()  # Store all IDs and its types
        self.nodeItems = dict()
        self.scene = QtWidgets.QGraphicsScene(self)

        self.scene.setSceneRect(QtCore.QRectF(-500, -500, 1000, 1000))
        self.set_pixmap('images/draw/threeHumans.png')

        for cam in self.graph.type_map_debug:
            self.all_types = {**self.graph.type_map_debug[cam], **self.all_types}
            for n_type in self.graph.type_map_debug[cam].values():
                # if self.coordinates_for_node_type_center(n_type, cam) is None:
                #     continue
                x, y = self.coordinates_for_node_type(n_type, cam)
                if n_type == 'sb':
                    colour = QtCore.Qt.lightGray
                    node_radius = 30
                elif n_type == 'b':
                    colour = QtCore.Qt.lightGray
                    node_radius = 25
                else:
                    colour = QtCore.Qt.black
                    node_radius = 7

                item = self.scene.addEllipse(x - node_radius, y - node_radius,
                                             node_radius*2, node_radius*2,
                                             brush=colour)
                self.nodeItems[n_type] = item
        self.setScene(self.scene)
        for cam in self.graph.type_map_debug:
            edges = self.graph.edges_debug[cam]
            for edge in edges:

                edge_a = self.all_types[edge[0]]
                edge_b = self.all_types[edge[1]]

                # if edge_a == 'b' or edge_b == 'b':
                #     continue
                if edge_a == edge_b:  # No self edges
                    continue

                ax, ay = self.coordinates_for_node_type(edge_a, cam)
                bx, by = self.coordinates_for_node_type(edge_b, cam)
                pen = QtGui.QPen()

                colour, width = self.type_to_colour_width(edge_a,
                                                          edge_b)
                pen.setColor(colour)
                pen.setWidth(width)

                self.scene.addLine(ax, ay, bx, by, pen=pen)

            for n_type in self.graph.type_map_debug[cam].values():
                if self.coordinates_for_node_type(n_type, cam) is None:
                    continue
                c = self.coordinates_for_node_type(n_type, cam)
                item = self.scene.addText(n_type)
                item.setDefaultTextColor(QtCore.Qt.magenta)
                item.setPos(*c)
                self.nodeItems[n_type] = item
            # print(len(self.graph))

    @staticmethod
    def coordinates_for_node_type(t, cam):
        mapping = {
            'sb': (-95, -400),
            'b': (-95, -250),
            'n': (6, -185),
            'le': (32, -200),
            'ley': (14, -201),
            'ls': (73, -116),
            'lel': (85, -25),
            'lw': (106, 48),
            'lh': (48, 35),
            'lk': (34, 193),
            'la': (28, 317),
            're': (-26, -200),
            'rey': (-7, -201),
            'rs': (-63, -116),
            'rel': (-75, -25),
            'rw': (-91, 48),
            'rh': (-35, 35),
            'rk': (-24, 193),
            'ra': (-18, 317)
        }
        ret = mapping[t]
        if t == 'sb':
            return ret
        return (ret[0]+311*(cam-1)-312, ret[1])

    # def cam_coordinates(self, cam):
    #     mapping = {
    #
    #     }


    @staticmethod
    def type_to_colour_width(type1, type2):
        if type1 == 'sb' or type2 == 'sb':
            colour = QtCore.Qt.lightGray
            width = 5
        elif type1 == 'b' or type2 == 'b':
            colour = QtCore.Qt.lightGray
            width = 1
        elif type1 == 'n' or type2 == 'n':  # The only special case, no left or right
            colour = QtCore.Qt.black
            width = 5
        else:
            if type1[1:] == type2[1:]:
                colour = QtCore.Qt.green
                width = 1.5
            else:
                colour = QtCore.Qt.black
                width = 5

        return colour, width

    @staticmethod
    def node_type(n_type, graph):
        labels = graph.get_node_types_one_hot()
        mapping = dict.fromkeys(parts, 0)
        for idx, key in enumerate(mapping.keys()):
            mapping[key] = labels[idx]

        return mapping.get(n_type, None)

    def closest_node_view(self, event_x, event_y):
        x_mouse = (event_x - 493)
        y_mouse = (event_y - 493)
        # print(str(x_mouse) + ', ' + str(y_mouse))
        closest_node = 0
        old_dist = 10000
        for cam in self.graph.type_map_debug:
            for n_id in self.graph.type_map_debug[cam]:
                key = self.all_types[n_id]
                x, y = self.coordinates_for_node_type(key, cam)
                dist = abs(x - x_mouse) + abs(y - y_mouse)
                if dist < old_dist:
                    old_dist = dist
                    closest_node = n_id
        return closest_node

    def set_pixmap(self, pixmap_path):
        pixmap = QtGui.QPixmap(pixmap_path)
        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        pixmap_item.setPos(-455, -500)
        pixmap_item.setScale(1.)
        self.scene.addItem(pixmap_item)
        #  self.scene.addRect(-30, -30, 60, 60, pen=QtGui.QPen(QtCore.Qt.white), brush=QtGui.QBrush(QtCore.Qt.white))

    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            closest_node_id = self.closest_node_view(event.x()-7, event.y()-7)
            n_type = self.all_types[closest_node_id]
            if n_type is None:
                print('Not valid label')
            self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(self.node_type(n_type, self.graph) + ' ' + str(closest_node_id)))

            # print(features)
            features = self.graph.features[closest_node_id]
            one_hot = [str(int(x)) for x in features[0:len(self.graph.get_node_types_one_hot())]]
            rest = ['{:1.3f}'.format(x) for x in features[len(self.graph.get_node_types_one_hot()):(len(features) + 1)]]
            features_format = one_hot + rest

            for idx, feature in enumerate(features_format, 1):
                self.table.setItem(idx, 0, QtWidgets.QTableWidgetItem(feature))

            return True
        return False


if __name__ == '__main__':
    file = open(sys.argv[1], "rb")
    scenarios = json.loads(file.read())['data_set']

    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    else:
        start = 0

    view = MainClass(scenarios, start)

    exit_code = app.exec_()
    sys.exit(exit_code)

