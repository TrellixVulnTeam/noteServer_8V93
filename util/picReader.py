#!usr/bin/python
# coding=utf-8
import numpy as np
import cv2 as cv
import threading
import time
import os
import schedule
from config import cameraConfig as cc
from config import picConfig as pc


class picManager:
    recentPic = {}
    blkPic_dict = {}
    pptPic_dict = {}

    def __init__(self, interval):
        self.blkBoard_mutex = threading.Lock()
        self.blkBoard_listMutex = threading.Lock()

        self.pptPic_mutex = threading.Lock()
        self.pptPic_listMutex = threading.Lock()

        self.interval = interval

    def getBlkboard_pic(self, reqTime):
        image = b""
        if abs(int(self.recentPic["blkBoard"]) - int(reqTime)) < self.interval:
            picTime = self.recentPic["blkBoard"]

            '''read file with file mutex
            '''
            if self.blkBoard_mutex.acquire(cc.INTERVAL):
                with open(pc.PIC_LOCAL["1"] + picTime + ".jpeg", "rb") as f:
                    image = f.read()
                self.blkBoard_mutex.release()

        else:  # therer is no useful pic,then capture a new pic
            nowTime = str(int(time.time()))
            cap = cv.VideoCapture(cc.CAMERA_NUM)  # craete videoCapture object
            res, frame = cap.read()  # capture a new pic
            if res:  # change recentPic's time
                '''write file with file mutex
                '''
                if self.blkBoard_mutex.acquire(cc.INTERVAL):
                    cv.imwrite(pc.PIC_LOCAL["1"] + nowTime + ".jpeg", frame)
                    self.blkBoard_mutex.release()

                self.recentPic["blkBoard"] = nowTime
            self.blkPic_dict[nowTime] = nowTime  # add a new pic name to blkPic_list
            cap.release()
            with open(pc.PIC_LOCAL["1"] + nowTime + ".jpeg", "rb") as f:
                image = f.read()
        return image

    def getPPT_pic(self, reqTime):
        pass

    def DeleteUseless_pic(self):
        for key in self.blkPic_dict.keys():
            value = self.blkPic_dict[key]
            nowTime = int(time.time())
            if int(value) + pc.DELETE_INTERVAL < nowTime:
                os.remove(pc.PIC_LOCAL["1"] + value + ".jpeg")
                print "remove",value
                self.blkPic_dict.pop(key)
        for key in self.pptPic_dict.keys():
            nowTime = int(time.time())
            value = self.pptPic_dict[key]
            if int(value) + pc.DELETE_INTERVAL < nowTime:
                os.remove(pc.PIC_LOCAL["1"] + value + ".jpeg")
                self.pptPic_dict.pop(key)

'''
regist pic type to picManager
'''

def registerPic_type(type):
    picManager.recentPic[type] = 0


registerPic_type("blkBoard")  # black board pic type
registerPic_type("ppt")  # ppt pic type
picManager = picManager(cc.INTERVAL)

'''
TIMER DELETOR
in order to delete useless pic from pic folder
'''


def TD():
    global picManager
    #schedule.every(pc.DELETE_INTERVAL).seconds.do(picManager.DeleteUseless_pic)
    while True:
        #schedule.run_pending()
        picManager.DeleteUseless_pic()
        time.sleep(pc.DELETE_INTERVAL)
        print "---run delete---"

'''
downloadPic api for appRouter
'''


def downloadPic(type, reqTime):
    if type == "1":
        return picManager.getBlkboard_pic(reqTime)
    elif type == "2":
        return picManager.getPPT_pic(reqTime)
