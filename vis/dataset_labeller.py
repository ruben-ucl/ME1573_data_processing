# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:14:30 2022

@author: Ruben
"""

import sys, functools, os, glob, h5py, pywt, traceback
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from pathlib import Path
from time import sleep

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, printProgressBar

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("CWT labeller")
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        
        # Create and connect widgets
        self.btnStart = QPushButton('Load data')
        
        self.mainFig = plt.figure(frameon=False)
        self.mainCanvas = FigureCanvas(self.mainFig)
        
        btnRow = QGroupBox()
        btnRowLayout = QHBoxLayout()
        self.buttons = {}
        for label in ['Previous', 'Label 0', 'Label 1', 'Next']:
            self.buttons[label] = QPushButton(label)
            btnRowLayout.addWidget(self.buttons[label])
        btnRow.setLayout(btnRowLayout)
        
        self.progressBar = QProgressBar()
        
        self.readout = QLineEdit()
        self.readout.setReadOnly(True)
        
        # self.overviewFig = plt.figure()
        # self.overviewCanvas = FigureCanvas(self.overviewFig)
        
        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.btnStart)
        layout.addWidget(self.mainCanvas)
        layout.addWidget(btnRow)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.readout)
        # layout.addWidget(self.overviewCanvas)
        self.centralWidget.setLayout(layout)
        
        
    def update_progress(self, perc, trackid):
        self.progressBar.setValue(perc)
        self.set_readout_text(f'Reading {trackid}')
        
    def grey_out(self):
        self.btnStart.setEnabled(False)
        
    def set_readout_text(self, text):
        self.readout.setText(text)

class Controller(QObject):
    sendFPath = pyqtSignal(float)  # Signal for sending filepath to read to worker thread
    sendFigure = pyqtSignal(object)    # Signal for sending device objects to the worker thread
    updateReadout = pyqtSignal(str)
    
    def __init__(self, view):
        super().__init__()
        print(f'Controller running on thread: {int(QThread.currentThreadId())} (main)')
        self.view = view   # Define the view for access from within the Controller object
        self.connect_signals()
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        # Variables to track navigation through datasets
        self.index = -1
        self.nFiles = 0
        # Window length for CWT sections
        self.windowLength = 1 # ms
        # Margins from start and end of laser scan respectively to exclude
        # self.windowLimits = [0, 0] # ms
        # Define file locations
        self.folder = get_paths()['hdf5']
        self.dir0 = Path(self.folder, 'CWT_labelled', '0')
        self.dir1 = Path(self.folder, 'CWT_labelled', '1')
        for d in [self.dir0, self.dir1]:
            if not os.path.exists(d):
                os.makedirs(d)
    
    def connect_signals(self):
        self.view.btnStart.clicked.connect(self.get_data)
        self.view.btnStart.clicked.connect(self.view.grey_out)
        self.updateReadout.connect(self.view.set_readout_text)
        self.view.btnStart.clicked.connect(lambda: self.view.set_readout_text('Button clicked'))
        for bt in ['Label 0', 'Label 1', 'Next']:
            self.view.buttons[bt].clicked.connect(lambda: self.navigate('+'))
        self.view.buttons['Previous'].clicked.connect(lambda: self.navigate('-'))
        
        
    def get_data(self):
        print('Initialising Worker to read data')
        worker = Worker(self.read_files)
        worker.signals.progress.connect(self.view.update_progress)
        worker.signals.output.connect(self.keep_data)
        worker.signals.finished.connect(lambda: self.view.set_readout_text('Done'))
        # Execute
        self.threadpool.start(worker)
            
    def read_files(self):
        files = sorted(glob.glob(f'{self.folder}/*.hdf5'))[:4]
        self.nFiles = len(files)
        print(f'Reading {self.nFiles} files from \'{self.folder}\'')
        group, time, series, colour = ('AMPM', 'Time', 'Photodiode1Bits', 'r')
        # self.sendProgress.connect(self.view.update_progress)
        
        data = {'trackid': [], 't': [], 'PD': [], 'xray': []} 
        for i, filepath in enumerate(files):
            trackid = Path(filepath).name[:7]
            data['trackid'].append(trackid)
            
            with h5py.File(filepath, 'r') as file:
                data['t'].append(np.array(file[f'{group}/{time}'])[500:-500])
                data['PD'].append(np.array(file[f'{group}/{series}'])[500:-500])
                data['xray'].append(np.array(file['bs-f40'])[-1])
            
        df = pd.DataFrame(data)
        return df
    
    def navigate(self, direction):
        try:
            if direction == '+':
                if self.index >= self.nFiles - 1:
                    raise IndexError
                else:
                    self.index += 1
            elif direction == '-':
                if self.index <= 0:
                    raise IndexError
                else:
                    self.index -= 1
            row = self.data.iloc[self.index]
            trackid = row['trackid']
            print(trackid)
            worker = Worker(self.cwt, row)
            worker.signals.output.connect(self.plot)
            worker.signals.finished.connect(lambda: self.view.set_readout_text('Done'))
            self.threadpool.start(worker)
        except IndexError:
            print('No more files')
    
    def cwt(self, data):
        s = data['PD']
        t = data['t']
        scales = np.logspace(1, 7, num=256, base=2, endpoint=True)
        wavelet = "cmor1.5-1.0"
        samplingPeriod = round(t[1]-t[0], 9)
        samplingRate = round(1/samplingPeriod, 7)
        
        cwtmatr, freqs = pywt.cwt(s, scales, wavelet, samplingPeriod=samplingPeriod)
        cwtmatr = np.abs(cwtmatr[:-1, :-1])
        return((t, freqs, cwtmatr))
    
    def keep_data(self, data):
        self.data = data
    
    def plot(self, data):
        t, freqs, cwtmatr = data
        self.view.mainFig.clear()
        ax = plt.Axes(self.view.mainFig, [0., 0., 1., 1.])
        ax.set_axis_off()
        tAx, fAx = np.meshgrid(t*1000, freqs/1000)
        pcm = ax.pcolormesh(tAx, fAx, cwtmatr, cmap='jet', vmin=0, vmax=200)
        ax.set_yscale('log', base=2)
        ax = self.view.mainFig.add_axes(ax)
        self.view.mainCanvas.draw()

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        print(f'Worker instance running on thread: {int(QThread.currentThreadId())}')
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.output.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done  
            print('Worker finished\n')

class WorkerSignals(QObject):
    # Defines the signals available from a running worker thread.
    finished = pyqtSignal()
    output = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(tuple)
 

def main():
    Gui = QApplication(sys.argv)
    # Define Qt stylesheet for GUI
    Gui.setStyleSheet("""QLabel         {font-size: 9pt;}
                         QPushButton    {font-size: 9pt;}
                         QLineEdit      {font-size: 9pt;}
                         QRadioButton   {font-size: 9pt;}
                         QComboBox      {font-size: 9pt;}
                         QTabWidget     {font-size: 9pt;}
                         QGroupBox      {font: bold 9pt; color: gray}
                         QToolButton    {font-size: 9pt;}
                         """)
    # Gui.setAttribute(Qt.AA_EnableHighDpiScaling, True) # Enable highdpi scaling
    # Gui.setAttribute(Qt.AA_UseHighDpiPixmaps, True) # Use highdpi icons
    view = Window() # Define and then show GUI window
    view.show()
    Controller(view) # Initialise controller with access to the view
    sys.exit(Gui.exec())
    
if __name__ == '__main__':
    main()
