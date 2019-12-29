import matplotlib.pyplot as plt
import numpy as np
import time
import time, random
import math
import serial
from collections import deque
import scipy.signal as signal
from numpy.fft import fft,fftshift,ifft,fftfreq
from scipy.signal import find_peaks
import tkinter as tk

#Display loading 
class PlotData:
    def __init__(self, max_entries=30):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)

#initial
fig, (ax,ax2,ax3,ax4) = plt.subplots(4,1)
line,  = ax.plot(np.random.randn(100))
line2, = ax2.plot(np.random.randn(100))
line3, = ax3.plot(np.random.randn(100))
line4, = ax4.plot(np.random.randn(100))
plt.show(block = False)
plt.setp(line2,color = 'r')


PData= PlotData(500)
PData2= PlotData(500)
PData3= PlotData(500)
PData4= PlotData(500)
ax.set_ylim(-10,10)
ax2.set_ylim(-3,8)
ax3.set_ylim(0,500)
ax4.set_xlim(-2,2)
ax4.set_ylim(-2,2)

angle = np.linspace(-np.pi, np.pi, 50)
cirx = np.sin(angle)
ciry = np.cos(angle)

plt.xlabel('Real')
plt.ylabel('Imag')

# plot parameters
print ('plotting data...')
# open serial port
strPort='com4'
ser = serial.Serial(strPort, 115200)
ser.flush()

start = time.time()
temp = deque(maxlen=20)

sigma = 3
filter_size = 6*sigma+1
gauss = signal.gaussian(filter_size,sigma)
sum = np.sum(gauss)
gauss = gauss/sum
fs = 500
cut = 10/(0.5*fs)
bk = np.ones(20)/20
f = fftshift(fftfreq(fs, 1/fs))
line4.set_xdata(cirx)
line4.set_ydata(ciry)
plt.plot(np.real(np.roots(bk)), np.imag(np.roots(bk)), 'o', markersize=10)
b1, a1 = signal.butter(5,cut,'lowpass')
count = 0
num = 50
tnow =0


while True:
    
    for ii in range(10):

        try:
            data = float(ser.readline())
            temp.append(data)
            PData.add(time.time() - start, data-np.mean(temp))
        except:
            pass
    #PData2.axis_y=signal.lfilter(b, 1, PData.axis_y)
    ax.set_xlim(PData.axis_x[0], PData.axis_x[0]+5)
    ax2.set_xlim(PData.axis_x[0], PData.axis_x[0]+5)
    ax3.set_xlim(-30, 30)
    line.set_xdata(PData.axis_x)
    line.set_ydata(PData.axis_y)
    if(len(PData.axis_x)>=500): 
        #data2 = fft(PData.axis_y)
        #PData2.axis_y = np.convolve(PData.axis_y, bk, 'same')
        #datax=signal.lfilter(b, 1, PData.axis_y)
        #datas = np.convolve(PData.axis_y, gauss, 'same')
        #datas=signal.lfilter(bk, 1, PData.axis_y)
        #datax=signal.lfilter(bk, 1, datas)
        datas = signal.lfilter(b1,a1,PData.axis_y)
        #datas = np.convolve(PData.axis_y, bk, 'same')
        PData2.axis_y = np.convolve(datas, gauss, 'same')
        #PData2.axis_y = ilp_filter(PData.axis_y, cutoff, fs)
        if PData.axis_x[0]>=tnow:
            tnow+=0.5
            peaks, _ = find_peaks(PData2.axis_y,height=1, prominence=1)
            if len(peaks)>=3 and len(peaks)<=15:
                num += len(peaks)
                bpm = (6*num)/(PData.axis_x[0]+5)
                if tnow%1.5==0: 
                    print('現在心率:'+str(bpm)+'(bpm)')
            else:
                print('錯誤')
        #times = PData.axis_x[0]+5
        #countf = count/times
        line2.set_xdata(PData.axis_x)
        line2.set_ydata(PData2.axis_y)
        line3.set_xdata(f)
        line3.set_ydata(abs(fftshift(fft(PData2.axis_y))))
        fig.canvas.draw()
        fig.canvas.flush_events()
window= tk.Tk()
start= mclass (window)
root.mainloop()

        