from PyQt5.QtCore    import *
from PyQt5.QtGui     import *
from PyQt5.QtWidgets import *

import sys
import utils_q14 as utils

__appname__ = "2021 Opencvdl Hw2 "

class windowUI(object):
    """
    Set up UI
    please don't edit
    """
    def setupUI(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(480,640)
        MainWindow.setWindowTitle(__appname__)

        # 1. Background Subtraction
        BG_subtraction_Group = QGroupBox("1. Background Subtraction")
        group_V1_vBoxLayout = QVBoxLayout(BG_subtraction_Group)

        self.button1_1 = QPushButton("1.1 Background Subtraction")
        group_V1_vBoxLayout.addWidget(self.button1_1)

        # 2. Optical Flow Group
        Optical_Flow_Group = QGroupBox("2. Optical Flow")
        group_V2_vBoxLayout = QVBoxLayout(Optical_Flow_Group)

        self.button2_1 = QPushButton("2.1 Preprocessing")
        self.button2_2 = QPushButton("2.2 Video tracking")
        group_V2_vBoxLayout.addWidget(self.button2_1)
        group_V2_vBoxLayout.addWidget(self.button2_2)
        
        
        # 3. Perspective Transform Group
        Perspective_Transform_Group = QGroupBox("3. Perspective Transform")
        group_V3_vBoxLayout = QVBoxLayout(Perspective_Transform_Group)

        self.button3_1 = QPushButton("3.1 Perspective Transform")
        group_V3_vBoxLayout.addWidget(self.button3_1)

        # 4. PCA Group
        PCA_Group = QGroupBox("4. PCA")
        group_V4_vBoxLayout = QVBoxLayout(PCA_Group)

        self.button4_1 = QPushButton("4.1 Image Reconstruction")
        self.button4_2 = QPushButton("4.2 Compute The Reconstruction Error")
        group_V4_vBoxLayout.addWidget(self.button4_1)
        group_V4_vBoxLayout.addWidget(self.button4_2)   

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        vLayout = QVBoxLayout()
        vLayout.addWidget(BG_subtraction_Group)
        vLayout.addWidget(Optical_Flow_Group)
        vLayout.addWidget(Perspective_Transform_Group)
        vLayout.addWidget(PCA_Group)
        self.centralwidget.setLayout(vLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(MainWindow)

class MainWindow(QMainWindow, windowUI):

    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUI(self)
        self.initialValue()
        self.buildUi()

    def buildUi(self):
        self.button1_1.clicked.connect(self.bg_subtraction)

        self.button2_1.clicked.connect(self.preprocessing)
        self.button2_2.clicked.connect(self.video_tracking)

        self.button3_1.clicked.connect(self.perspective_transform)

        self.button4_1.clicked.connect(self.image_reconstruction)
        self.button4_2.clicked.connect(self.compute_error)  
    
    def initialValue(self):
        self.tracking = utils.Q2(r"./Q2_Image/optical_flow.mp4")
        self.pca = utils.Q4(r"./Q4_Image", 28)
        pass

    def bg_subtraction(self):
        path = r"./Q1_Image/traffic.mp4"
        utils.Q1(path)
        pass

    def preprocessing(self):
        self.tracking.initial()
        self.tracking.processing()
        pass

    def video_tracking(self):
        self.tracking.initial()
        self.tracking.processing(False)
        pass

    def perspective_transform(self):
        utils.Q3(r"./Q3_Image/perspective_transform.mp4", r"./Q3_Image/logo.png")
        pass

    def image_reconstruction(self):
        self.pca.imageReconstruction()
        pass

    def compute_error(self):
        self.pca.reconstructionErrorComputing()
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(500, 150, 500, 300)
    window.show()
    sys.exit(app.exec_())