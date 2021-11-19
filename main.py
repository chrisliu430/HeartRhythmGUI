from PyQt5 import QtWidgets, QtGui
from GUI import Ui_MainWindow
import sys
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = self.ui.ModelSelector.currentText()
        self.ui.OpenFileBtn.clicked.connect(self.OpenFile)
        self.ui.AnalysisBtn.clicked.connect(self.Analysis)
        self.pathFile = ""
        self.fileName = ""
        self.resultHashTable = {
            "Normal_Abnormal": ["Abnormal", "Normal"],
            "Normal_VSD": ["VSD", "Normal"],
            "Normal_ASD": ["ASD", "Normal"],
            "VSD_ASD": ["VSD", "ASD"]
        }
        
    
    def OpenFile(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        self.pathFile, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Select Analysis File", "","Wavlet Files(*.wav)", options=options)
        file = self.pathFile.split("/")
        self.fileName = file[len(file) - 1]
        if self.fileName:
            self.ui.FileLocationLabel.setText(self.fileName)

    def Analysis(self):
        self.model = self.ui.ModelSelector.currentText().split(" / ")
        selectModel = self.model[0] + "_" + self.model[1]
        if (selectModel == 'Normal_Abnormal'):
            heartSoundResult = subprocess.check_output(["python3", "models/ModelPredict.py", self.pathFile, self.fileName.strip(".wav"), selectModel])
        elif (selectModel == 'Normal_VSD'):
            heartSoundResult = subprocess.check_output(["python3", "models/model_predict_VSD.py", self.pathFile, self.fileName.strip(".wav"), selectModel])
        elif (selectModel == 'Normal_ASD'):
            heartSoundResult = subprocess.check_output(["python3", "models/model_predict_ASD.py", self.pathFile, self.fileName.strip(".wav"), selectModel])
        subprocess.check_output(["python3", "models/Segmentation_plt.py", self.pathFile, self.fileName.strip(".wav")])
        result = int(heartSoundResult.decode('utf-8'))
        resultText = self.resultHashTable[selectModel][result]
        if (resultText == 'Normal'):
            self.ui.ResultLabel.setText(resultText)
            self.ui.ResultLabel.setStyleSheet('color:green')
        else:
            self.ui.ResultLabel.setText(resultText)
            self.ui.ResultLabel.setStyleSheet('color:red')
        # heartChart = mpimg.imread('./images/HeartSound/' + self.fileName.strip(".wav") + ".png")
        # segmented = mpimg.imread('./images/Segmented/' + self.fileName.strip(".wav") + ".png")
        # plt.subplot(2, 1, 1)
        # plt.imshow(heartChart)
        # plt.subplot(2, 1, 2)
        # plt.imshow(segmented)
        # plt.show()
        self.ui.HeartChart.setPixmap(QtGui.QPixmap('./images/HeartSound/' + self.fileName.strip(".wav") + '.png'))
        self.ui.HeartChart.setScaledContents(True)
        self.ui.SegmentedChart.setPixmap(QtGui.QPixmap('./images/Segmented/' + self.fileName.strip(".wav") + '.png'))
        self.ui.SegmentedChart.setScaledContents(True)

if __name__ == '__main__':
    try:
        os.mkdir("./images/")
        os.mkdir("./images/HeartSound/")
        os.mkdir("./images/Segmented/")
    except OSError:
        print()
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())