#!/usr/bin/env python
__author__ = 'Jakub Kvita & Jan Bednarik'

import threading
import numpy as np
import cv2
import cv2.cv as cv
import datetime

class Video(threading.Thread):
    def __init__(self):
        super(Video, self).__init__()

        self.play = True
        self.exit = False
        self.waitPerFrameInMillisec = 1
        self.filename = ''

    # no.1
        self.grayscale = False
    # no.2
        self.invert = False
    # no.3
        self.histeql = False
    # no.4
        self.thresholding = False
        self.threshold = 125
    # no.5
        self.reducecolors = False
        self.reducechannelvalues = 2
    # no.5
        self.median = False
        self.mediansize = 3
    # no.6
        self.blur = False
        self.blurkernelsize = 3
        self.blurkernel = []
    # no.7
        self.sharpen = False
        self.sharpenkernelsize = 3
        self.sharpenkernel = []
    # no.8
        self.edges = False
        self.edgeskernel = []
    # no.9
        self.canny = False
        self.cannynumber = 10
        self.cannynumber2 = 30

###############generovani blur kernelu
        for i in range(3,32,2):
            self.blurkernel.insert(i/2-1, cv2.getGaussianKernel(i,0) * cv2.transpose(cv2.getGaussianKernel(i,0)))
        # print("Blur Kernel my")
        # print(self.blurkernel)
###############generovani sharpen kernelu
        # for i in range(3,32,2):
        #     self.sharpenkernel.insert(i, cv2.Laplacian(self.blurkernel[i/2-1], cv2.CV_64F))
        #     self.sharpenkernel[i/2-1] *= -1
        for i in range(3,32,2):
            self.sharpenkernel.insert(i,self.blurkernel[i/2-1].copy())
            self.sharpenkernel[i/2-1][i/2][i/2] -= 2.
            self.sharpenkernel[i/2-1] *=- 1
        # print("Sharpen Kernel my")
        # print(self.sharpenkernel)
#######################generovani edge kernelu
        self.edgeskernel = np.array([[ -1.,  -1.,  -1.],
                                    [-1.,  8.,  -1.],
                                    [-1.,  -1.,  -1.]])
        # print self.edgeskernel

    def setFile(self, filename):
        if filename == 0 and self.filename == 0:
            return
        self.filename = filename
        self.cap = cv2.VideoCapture(self.filename)
        self.fps = self.cap.get(cv.CV_CAP_PROP_FPS)
        self.waitPerFrameInMillisec = int(1 / self.fps * 1000 / 1) if self.fps > 0 else 15
        # print 'Frame Rate = ', self.fps, ' frames per sec'
        # print 'Wait between frames = ', self.waitPerFrameInMillisec, ' ms'
        self.play = True

    def run(self):
        frameorig = []
        while not self.exit:
            activefilters = 0
    ####
            # before = datetime.datetime.now()
    ####
            if self.cap.isOpened():

                if self.play:
                    #pokud hrajeme dal nactu dalsi, jinak porad tocim ten stary
                    ret, frameorig = self.cap.read()
                    frame = frameorig
                else:
                    frame = frameorig

############################################ FILTRY
                if self.grayscale:
                    activefilters+=1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.invert:
                    activefilters+=1
                    frame = (255-frame)

                if self.histeql:
                    activefilters+=1
                    if self.grayscale:
                        frame = cv2.equalizeHist(frame)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                        channels = cv2.split(frame)
                        channels[0] = cv2.equalizeHist(channels[0])
                        frame = cv2.merge(channels)
                        frame = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR)

                if self.thresholding:
                    activefilters+=1
                    retval, frame = cv2.threshold(frame,self.threshold,255,cv2.THRESH_BINARY)

                if self.reducecolors:
                    activefilters+=25
                    frame = frame.astype(np.float16)
                    frame = (frame / (255./(self.reducechannelvalues-1))).round() * (255/(self.reducechannelvalues-1))
                    frame = frame.astype(np.uint8)

                if self.median:
                    activefilters+=self.mediansize*2
                    frame = cv2.medianBlur(frame,self.mediansize)

                if self.blur:
                    activefilters+=self.blurkernelsize*2
                    frame = cv2.filter2D(frame, -1, self.blurkernel[self.blurkernelsize/2-1])

                if self.sharpen:
                    activefilters+=self.sharpenkernelsize*2
                    frame = cv2.filter2D(frame, -1, self.sharpenkernel[self.sharpenkernelsize/2-1])

                if self.edges:
                    activefilters+=1
                    frame = cv2.filter2D(frame, -1, self.edgeskernel)

                if self.canny:
                    activefilters+=1
                    frame = cv2.Canny(frame, self.cannynumber, self.cannynumber2)
##########################################

                cv2.imshow("Video", frame)
        ####
                # after = datetime.datetime.now()
                # delta = after - before
                # print delta,"   ",delta.microseconds/1000
        ###
                wait = self.waitPerFrameInMillisec-activefilters
                # print wait
                cv2.waitKey(wait if wait>0 else 1)
                # cv2.waitKey(self.waitPerFrameInMillisec-int(delta.microseconds/10000.))
                # cv2.waitKey(1)
            else:
                #jestli neni nic otevreno tak cekam az si neco vybere a pusti
                cv2.waitKey(200)

        self.cap.release()
        cv2.destroyAllWindows()