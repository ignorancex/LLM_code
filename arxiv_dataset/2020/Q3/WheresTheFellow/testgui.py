import sys
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem
from PySide2.QtGui import *
from PySide2.QtCore import *
import numpy as np
import matplotlib.pyplot as plt
import json, sys
import math
import graph_generator
import trackerapi
import math
import pickle



class TestGUI(QMainWindow):
    def __init__(self):
        self.app = QApplication(sys.argv)
        ui_file = QFile("testgui.ui")
        ui_file.open(QFile.ReadOnly)

        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        self.scene_gt = QGraphicsScene()
        self.scene_gt.setSceneRect(-2000, -3000, 4000, 6000)

        self.view_gt = QGraphicsView(self.scene_gt, self.window.scene_gt)
        self.view_gt.scale( -0.1, 0.1 );

        self.view_gt.resize(self.window.scene_gt.geometry().width(), self.window.scene_gt.geometry().height())

        self.personPos_gt = self.scene_gt.addEllipse(QRectF(-200,-200, 400, 400), QPen(QColor("LightGreen")), QBrush(QColor("LightGreen")))
        self.personPos_gt.setFlag(QGraphicsItem.ItemIsMovable)
        self.personPos_gt.setPos(0, 0)
        self.personAng_gt = self.scene_gt.addRect(QRectF(-10,0, 20, 300), QPen(QColor("Green")), QBrush(QColor("Green")))
        self.personAng_gt.setFlag(QGraphicsItem.ItemIsMovable)
        self.personAng_gt.setPos(0, 0)


        self.scene_cc = QGraphicsScene()
        self.scene_cc.setSceneRect(-2000, -3000, 4000, 6000)

        self.view_cc = QGraphicsView(self.scene_cc, self.window.scene_cc)
        self.view_cc.scale( -0.1, 0.1 );

        self.view_cc.resize(self.window.scene_cc.geometry().width(), self.window.scene_cc.geometry().height())

        self.personPos_cc = self.scene_cc.addEllipse(QRectF(-200,-200, 400, 400), QPen(QColor("Red")), QBrush(QColor("Red")))
        self.personPos_cc.setFlag(QGraphicsItem.ItemIsMovable)
        self.personPos_cc.setPos(0, 0)
        self.personAng_cc = self.scene_cc.addRect(QRectF(-10,0, 20, 300), QPen(QColor("Black")), QBrush(QColor("Black")))
        self.personAng_cc.setFlag(QGraphicsItem.ItemIsMovable)
        self.personAng_cc.setPos(0, 0)

        self.scene_nn = QGraphicsScene()
        self.scene_nn.setSceneRect(-2000, -3000, 4000, 6000)

        self.view_nn = QGraphicsView(self.scene_nn, self.window.scene_nn)
        self.view_nn.scale( -0.1, 0.1 );

        self.view_nn.resize(self.window.scene_nn.geometry().width(), self.window.scene_nn.geometry().height())

        self.personPos_nn = self.scene_nn.addEllipse(QRectF(-200, -200, 400, 400), QPen(QColor("LightBlue")), QBrush(QColor("LightBlue")))
        self.personPos_nn.setFlag(QGraphicsItem.ItemIsMovable)
        self.personPos_nn.setPos(0, 0)
        self.personAng_nn = self.scene_nn.addRect(QRectF(-10, 0, 20, 300), QPen(QColor("Blue")), QBrush(QColor("Blue")))
        self.personAng_nn.setFlag(QGraphicsItem.ItemIsMovable)
        self.personAng_nn.setPos(0, 0)

        self.loadData(sys.argv[1])

        self.window.instantScrollBar.valueChanged.connect(self.changeInstant)

        self.it = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.compute)
        self.timer.start(150)


        self.window.show()

        r = self.app.exec_()
        sys.exit(r)

    def loadData(self, filename):
        test_dataset = graph_generator.CalibrationDataset(filename, 'run','1')
        with open(sys.argv[1], 'r') as f:
            raw = f.read()
        raw = list(raw)

        raws = ''.join(raw)
        data = json.loads(raws)['data_set']

        model = trackerapi.TrackerAPI('.', test_dataset)

        self.x_gt = []
        self.x_cc = []
        self.x_nn = []
        self.z_gt = []
        self.z_cc = []
        self.z_nn = []
        self.a_gt = []
        self.a_cc = []
        self.a_nn = []

        try:
            with open('kk', 'rb') as f:
                results = pickle.load(f)
        except:
            results = [x for x in model.predict()]
            with open('kk', 'wb') as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

        eX_cc = []
        eZ_cc = []
        eA_cc = []
        eX_nn = []
        eZ_nn = []
        eA_nn = []

        self.trajectory = []

        s = 0
        ang_prev = 0
        for i in range(int(len(results))):

            n_joints = 0
            for cam in range(len(data[i]['superbody'])):
               n_joints += len(data[i]['superbody'][cam]['joints'])

            if n_joints < 3:
                continue

            s += 1
            if s<2 or s%1 != 0:
                continue


            self.x_gt.append(data[i]['superbody'][0]['ground_truth'][0])
            self.z_gt.append(data[i]['superbody'][0]['ground_truth'][2])
            self.a_gt.append(data[i]['superbody'][0]['ground_truth'][3]*180./math.pi)

            self.trajectory.append(QPointF(self.x_gt[-1], self.z_gt[-1]))

            x_cc = 0
            z_cc = 0
            ncams = 0
            for cam in range(0, len(data[i]['superbody'])):
                x_cc += data[i]['superbody'][cam]['world'][0]
                z_cc += data[i]['superbody'][cam]['world'][2]
                ncams += 1

            self.x_cc.append(x_cc/ncams)
            self.z_cc.append(z_cc/ncams)

            s_cc = 0
            c_cc = 0
            ncams = 0
            for cam in range(0, len(data[i]['superbody'])):
                joints = data[i]['superbody'][cam]["joints"]
                if ("right_shoulder" in joints and "left_shoulder" in joints) or ("right_hip" in joints and "left_hip" in joints):
                    s_cc = math.sin(data[i]['superbody'][cam]['world'][3])
                    c_cc = math.cos(data[i]['superbody'][cam]['world'][3])
                    ncams += 1

            if ncams>0:
                a_cc = math.atan2(s_cc/ncams, c_cc/ncams)
                ang_prev = a_cc
            else:
                a_cc = ang_prev
                    
            self.a_cc.append(a_cc*180./math.pi)

            self.x_nn.append(results[i][0]*4000)
            self.z_nn.append(results[i][2]*4000)
            self.a_nn.append(math.atan2(results[i][3]/0.7, results[i][4]/0.7)*180./math.pi)
            eX_cc.append(abs(self.x_gt[-1]-self.x_cc[-1]))
            eZ_cc.append(abs(self.z_gt[-1]-self.z_cc[-1]))
            eAng = 180 - abs(abs(self.a_gt[-1]-self.a_cc[-1]) -180)
            if eAng < 0:
                eAng = 360 + eAng
            eA_cc.append(eAng)

            eX_nn.append(abs(self.x_gt[-1]-self.x_nn[-1].item()))
            eZ_nn.append(abs(self.z_gt[-1]-self.z_nn[-1].item()))
            eAng = 180 - abs(abs(self.a_gt[-1]-self.a_nn[-1]) -180)
            if eAng < 0:
                eAng = 360 + eAng
            eA_nn.append(eAng)

        array_err_z = np.array(eZ_nn)
        print("len array error", len(array_err_z))


#        self.window.eX_cc.display(np.array(eX_cc).mean())
#        self.window.eZ_cc.display(np.array(eZ_cc).mean())
#        self.window.eA_cc.display(np.array(eA_cc).mean())

#        self.window.eX_nn.display(np.array(eX_nn).mean())
#        self.window.eZ_nn.display(np.array(eZ_nn).mean())
#        self.window.eA_nn.display(np.array(eA_nn).mean())

        self.scene_gt.addPolygon(self.trajectory, QPen(QColor("Black")))
        self.scene_cc.addPolygon(self.trajectory, QPen(QColor("Black")))
        self.scene_nn.addPolygon(self.trajectory, QPen(QColor("Black")))

        self.window.instantScrollBar.setMaximum(len(self.a_gt)-1)


    def compute(self):
        if self.window.playButton.isChecked():
            if self.it >= len(self.x_gt):
                self.it=0
            self.movePerson()
            self.it += 1
            self.window.instantScrollBar.setValue(self.it)

    def changeInstant(self, instant):
        self.it = instant 
        self.movePerson()

    def movePerson(self):

        self.personPos_gt.setPos(self.x_gt[self.it], self.z_gt[self.it])
        self.personAng_gt.setPos(self.x_gt[self.it], self.z_gt[self.it])
        self.personAng_gt.setRotation(self.a_gt[self.it])

        self.personPos_cc.setPos(self.x_cc[self.it], self.z_cc[self.it])
        self.personAng_cc.setPos(self.x_cc[self.it], self.z_cc[self.it])
        self.personAng_cc.setRotation(self.a_cc[self.it])

        self.personPos_nn.setPos(self.x_nn[self.it], self.z_nn[self.it])
        self.personAng_nn.setPos(self.x_nn[self.it], self.z_nn[self.it])
        self.personAng_nn.setRotation(self.a_nn[self.it])



if __name__ == "__main__":

    TestGUI()


