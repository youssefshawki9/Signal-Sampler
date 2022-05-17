from scipy.fft import fft
import scipy.signal as sig
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
import pyqtgraph as pg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from numpy import *
import sys
import matplotlib
matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(
            r'try.ui', self)

        # initialize variables
        self.mag = 0
        self.freq = 1
        self.phase = 0

        self.sinusoidal = 0
        self.signalSum = 0
        self.signal = 0
        self.signalName = ""

        self.time = arange(0.0, 1.0, 0.001)

        # dictionary with signal variables
        self.signaldict = dict()

        # initialize variables for matplotlib plotting
        self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout1 = QtWidgets.QVBoxLayout()
        self.layout1.addWidget(self.canvas1)

        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout2 = QtWidgets.QVBoxLayout()
        self.layout2.addWidget(self.canvas2)

        self.canvas3 = MplCanvas(
            self.signalWidget1,  width=5, height=4, dpi=100)
        self.layout3 = QtWidgets.QVBoxLayout()
        self.layout3.addWidget(self.canvas3)

        self.canvas4 = MplCanvas(
            self.signalWidget2, width=5, height=4, dpi=100)
        self.layout4 = QtWidgets.QVBoxLayout()
        self.layout4.addWidget(self.canvas4)

        # button connections
        self.horizontalSlider.valueChanged.connect(lambda: self.plotHSlide())
        self.hideButton.clicked.connect(lambda: self.hide())
        self.open.clicked.connect(lambda: self.load())
        self.plot.clicked.connect(lambda: self.displaySig())
        self.add.clicked.connect(lambda: self.displaySum())
        self.remove.clicked.connect(lambda: self.signalRemove())
        self.signalCombobox.currentIndexChanged.connect(
            lambda: self.signalChoice())
        self.confirm.clicked.connect(lambda: self.signalConfirm())

        self.loaded = False

        self.graph = pg.PlotItem()
        pg.PlotItem.hideAxis(self.graph, 'left')
        pg.PlotItem.hideAxis(self.graph, 'bottom')

    # define signal using given parameters
    def signalParameters(self, magnitude, frequency, phase):
        omega = 2*pi*frequency
        theta = phase*pi/180
        return magnitude*sin(omega*self.time + theta)

    # signal plotter function
    def signalPlot(self, canvas, widget, layout, signal):
        canvas.axes.cla()
        canvas.axes.plot(self.time, signal)
        canvas.draw()
        widget.setCentralItem(self.graph)
        widget.setLayout(layout)

    # return signal function/values through data
    def getSignal(self):
        self.signalName = self.signalCombobox.currentText()
        self.valueList = self.signaldict[self.signalName]
        self.signal = self.signalParameters(
            self.valueList[0], self.valueList[1], self.valueList[2])

    # user-defined signal and plot
    def displaySig(self):
        self.mag = float(self.magnitude.text())
        self.freq = float(self.frequency.text())
        self.phase = float(self.phaseshift.text())

        if self.cosbtn.isChecked() == True:
            self.phase += 90

        self.name = self.plotname.text()
        self.signaldict[self.name] = self.mag, self.freq, self.phase
        self.signalCombobox.addItem(self.name)

        self.sinusoidal = self.signalParameters(
            self.mag, self.freq, self.phase)

        self.signalPlot(self.canvas1, self.sinoPlotter,
                        self.layout1, self.sinusoidal)

    # sum of generated signals
    def displaySum(self):
        self.signalSum += self.sinusoidal
        self.signalPlot(self.canvas2, self.sinoAdder,
                        self.layout2, self.signalSum)

    # signal name changed
    def signalChoice(self):
        if self.signalCombobox.count() == 0:
            raise ValueError('the combobox is empty cant be accessed')
        else:
            self.getSignal()
            self.signalPlot(self.canvas1, self.sinoPlotter,
                            self.layout1, self.signal)

    # remove selected signal
    def signalRemove(self):
        if self.signalCombobox.count() == 1:
            self.signalSum = [0]*(len(self.time))
            self.signaldict.clear()
            self.signalCombobox.clear()
        else:
            index = self.signalCombobox.currentIndex()
            self.getSignal()
            self.signalCombobox.removeItem(index)
            self.signalSum -= self.signal
            self.signaldict.pop(self.signalName, None)
        self.signalPlot(self.canvas2, self.sinoAdder,
                        self.layout2, self.signalSum)

    # confirm signal to main illustrator
    def signalConfirm(self):
        self.canvas3.axes.clear()
        self.canvas4.axes.clear()
        self.xData = self.time
        self.yData = self.signalSum
        max_freq = []
        for i in self.signaldict: 
            max_freq.append(self.signaldict[i][1])
        self.maxFreq = max(max_freq)
        if self.maxFreq < 2:
            self.horizontalSlider.setMinimum(1)
            self.horizontalSlider.setSingleStep(1)
            self.horizontalSlider.setPageStep(1)
        self.horizontalSlider.setValue(self.horizontalSlider.minimum())
        self.Plot()

    def load(self):
        self.canvas3.axes.clear()
        self.canvas4.axes.clear()
        self.fname1 = QFileDialog.getOpenFileName(
            None, "Select a file...", os.getenv('HOME'), filter="All files (*)")
        path1 = self.fname1[0]
        data1 = pd.read_csv(path1)
        self.yData = data1.values[:, 1]
        self.xData = data1.values[:, 0]
        self.horizontalSlider.setValue(self.horizontalSlider.minimum())
        if self.loaded == True:
            self.layout1.removeWidget(self.canvas3)
            self.layout2.removeWidget(self.canvas4)

        # Filtering
        FTydata = np.fft.fft(self.yData)
        FTydata = FTydata[0:int(len(self.yData)/2)]
        FTydata = abs(FTydata)
        maxpower = max(FTydata)
        noise = (maxpower/100)
        self.fmaxtuble = np.where(FTydata > noise)
        self.maxFreq = max(self.fmaxtuble[0])
        print(self.maxFreq)
        self.loaded = True
        self.Plot()

    def sinc_interp(self, x, s, u):

        if len(x) != len(s):
            raise ValueError('x and s must be the same length')

    # Find the period
        T = s[1] - s[0]

        sincM = np.tile(u, (len(s), 1)) - \
            np.tile(s[:, np.newaxis], (1, len(u)))
        y = np.dot(x, np.sinc(sincM/T))
        return y

    # Plotting the 
    def Plot(self):

        self.horizontalSlider.setMaximum(int(ceil(3*self.maxFreq)))

        # smapling the data
        resampledData = sig.resample(
            self.yData, self.horizontalSlider.value(), self.xData)

        sampledData = resampledData[0]
        sampledtime = resampledData[1]
        sampledtime[0]=self.xData[0]
        sampledData[0]=self.yData[0]
        sampledtime=np.append(sampledtime,[self.xData[-1]])
        sampledData=np.append(sampledData,[self.yData[-1]])

        #interpolatng on the new data 
        recontructedData = self.sinc_interp(sampledData, sampledtime, self.xData)

        # plotting the original signal and the sampled data as dots and the reconstructed signal 
        self.canvas3.axes.plot(self.xData, self.yData)
        self.canvas3.axes.plot(self.xData, recontructedData)
        self.canvas3.axes.scatter(sampledtime, sampledData, color='k', s=10)
        self.canvas3.draw()
        self.signalWidget1.setCentralItem(self.graph)
        self.signalWidget1.setLayout(self.layout3)

        # plotting the constructed data on the second graph
        self.canvas4.axes.plot(self.xData, recontructedData, color='r')
        self.canvas4.draw()
        self.signalWidget2.setCentralItem(self.graph)
        self.signalWidget2.setLayout(self.layout4)

    def plotHSlide(self):
        self.canvas4.axes.clear()
        self.canvas3.axes.clear()
        self.Plot()

    def hide(self):
        if self.signalWidget2.isHidden():
            self.signalWidget2.show()
        else:
            self.signalWidget2.hide()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
