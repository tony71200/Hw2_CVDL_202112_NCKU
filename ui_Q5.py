import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QLabel, QWidget, QLineEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QGroupBox, QMessageBox
from PyQt5.QtCore import Qt, QMetaObject

import cv2 as cv
import glob
import os
import numpy as np
from utils_Q5_DC import Q5_train, Q5_test
__appname__ = "2021 Opencvdl Hw1 Question 5"

class windowUI(object):
    """
    Set up UI
    please don't edit
    """
    def setupUI(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640,480)
        MainWindow.setWindowTitle(__appname__)


        # 5. VGG Test Group
        VGG_Group = QGroupBox("5. VGG Test")
        group_V1_vBoxLayout = QVBoxLayout(VGG_Group)

        self.button5_1 = QPushButton("5.1 Training 5 epochs")
        self.button5_2 = QPushButton("5.2 Show tensorboard")

        Test_Group = QGroupBox("5.3 Test")
        group_V1_5_vBoxLayout = QVBoxLayout(Test_Group)

        select_layout, self.edit_5_3 = self.edit_Text("Select image: ")
        group_V1_5_vBoxLayout.addLayout(select_layout)
        self.button5_3 = QPushButton("5.3 Test")
        group_V1_5_vBoxLayout.addWidget(self.button5_3)
        
        self.button5_4 = QPushButton("5.4 Augumentation Random-Eraser")

        group_V1_vBoxLayout.addWidget(self.button5_1)
        group_V1_vBoxLayout.addWidget(self.button5_2)
        
        group_V1_vBoxLayout.addWidget(Test_Group)
        group_V1_vBoxLayout.addWidget(self.button5_4)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        vLayout = QHBoxLayout()
        vLayout.addWidget(VGG_Group)
        self.centralwidget.setLayout(vLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(MainWindow)

    @staticmethod
    def edit_Text(title:str, unit = "", showUnit= False):
        hLayout = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setFixedWidth(60)
        unit_label = QLabel(unit)
        unit_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        unit_label.setFixedWidth(30)
        editText = QLineEdit("1")
        editText.setFixedWidth(50)
        editText.setAlignment(Qt.AlignRight)
        editText.setValidator(QIntValidator())

        hLayout.addWidget(title_label, alignment=Qt.AlignLeft)
        hLayout.addWidget(editText)
        if showUnit:
            hLayout.addWidget(unit_label)
        return hLayout, editText

class MainWindow(QMainWindow, windowUI):

    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUI(self)
        self.initialValue()
        self.buildUi()

    def buildUi(self):
        self.button5_1.clicked.connect(self.train)
        self.button5_2.clicked.connect(self.show_tensorboard)
        self.button5_3.clicked.connect(self.test)
        self.button5_4.clicked.connect(self.show_augumentation)
    
    def initialValue(self):
        self.train_model = Q5_train()
        self.train_model.load_train_dataset()

        self.test_model = Q5_test()
        self.test_model.load_test_dataset()
        pass

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))
    
    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def train(self):
        self.setEnabled(False)
        self.train_model.train()
        self.setEnabled(True)
        pass

    def show_tensorboard(self):
        path = r""
        if path:
            image = cv.imread(path)
            cv.imshow("Tensorboard", image)
            cv.waitKey()
        cv.destroyAllWindows()
        pass

    def test(self):
        index = int(self.edit_5_3.text())
        self.setEnabled(False)
        self.status("Please wait few minutes")
        QApplication.processEvents()
        self.test_model.predict(index)
        self.setEnabled(True)
        pass

    def show_augumentation(self):
        self.setEnabled(False)
        self.train_model.show_before_after("model/compareModel.txt")
        QApplication.processEvents()
        self.setEnabled(True)
        pass




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(500, 150, 300, 300)
    window.show()
    sys.exit(app.exec_())