#!usr/bin/python
# coding=utf-8
import numpy as np
import cv2 as cv
import threading
import numpy as np
import tensorflow as tf
import sys
import copy
import time
import os
sys.path.insert(0,"./util/model")

from config import cameraConfig as cc
from config import picConfig as pc
from model.object_detection.utils import label_map_util
from model.object_detection.utils import visualization_utils as vis_util

cap=None  # craete videoCapture object
handler=None
class todo(object):
    def __init__(self):
        self.PATH_TO_CKPT = './util/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = './util/tfrecord/blackboard_label_map.pbtxt'
        self.NUM_CLASSES = 1
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    boxes[0,:],
                    classes[0,:].astype(np.int32),
                    scores[0,:],
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
        #cv.namedWindow("detection", cv.WINDOW_NORMAL)
        #cv.imshow("detection", image)
        #cv.waitKey(0)
        return boxes[0][0]
class handlerObj():
    def __init__(self,todoObj,dilateTime=2,erodeTime=1,rmNoise_type=1,iterTime=5):
        self.rmNoise_type=rmNoise_type
        self.dilateTime=dilateTime
        self.erodeTime=erodeTime
        self.todoObj=todoObj
        self.iterTime=iterTime

        self.function={0:cv.GaussianBlur,1:cv.medianBlur}

    def originHandle(self,img,kernelSize=5):
        #remove noise
        #use medianBlur default
        originImg=img.copy()
        if self.rmNoise_type==0:
            tempImg=self.function[self.rmNoise_type](originImg,(kernelSize,kernelSize),0)
        else:
            tempImg=self.function[self.rmNoise_type](originImg,kernelSize)
        #cv.imshow("blur",tempImg)

        #dilate and erode
        time_s=time.time()
        kernel=np.ones((kernelSize,kernelSize),np.uint8)
        tempImg=cv.dilate(tempImg,kernel,self.dilateTime)
        tempImg=cv.erode(tempImg,kernel,self.erodeTime)
        #cv.imshow("de", tempImg)

        #grabcut
        mask=np.zeros(tempImg.shape[:2],np.uint8)
        bgdModel=np.zeros((1,65),np.float64)
        fgdModel=np.zeros((1,65),np.float64)
        #get Rectangle
        bottom,left,top,right=self.todoObj.detect(img)
        time_e=time.time()
        print "--------------detect&de:",str(time_e-time_s)

        height=len(tempImg)
        width=len(tempImg[0])
        blkHeight=int((top-bottom)*height)
        blkWidth=int((right-left)*width)
        rect=(int(left*width),int(bottom*height),blkWidth,blkHeight)
        #print rect
        #grabCut and transform into grayImage
        time_s = time.time()
        cv.grabCut(tempImg,mask,rect,bgdModel,fgdModel,self.iterTime,cv.GC_INIT_WITH_RECT)
        time_e=time.time()
        print "--------------grab:",str(time_e-time_s)
        cutMask=np.where(((mask==2)|(mask==0)),0,1).astype("uint8")
        tempImg = tempImg * cutMask[:, :, np.newaxis]
        gray = cv.cvtColor(tempImg, cv.COLOR_BGR2GRAY)

        #scale the image size in order to commit harrisCorner
        time_s=time.time()
        newColum=np.zeros(height)
        #add six colum
        for i in range(0,3,1):
            gray=np.insert(gray,0,values=newColum,axis=1)
        for i in range(0, 3, 1):
            gray=np.insert(gray,width+3,values=newColum,axis=1)
        #add six row
        newRow=np.zeros(width+6)
        for i in range(0,3,1):
            gray=np.insert(gray,0,values=newRow,axis=0)
        for i in range(0,3,1):
            gray=np.insert(gray,height+3,values=newRow,axis=0)
        time_e=time.time()
        gray = np.float32(gray)
        print "-----------insert:",str(time_e-time_s)
        time_s=time.time()
        dst = cv.cornerHarris(gray, 20, 7, 0.04)
        points=[]
        max=0.1*dst.max()
        for x in range(3,height+3,1):
            for y in range(3,width+3,1):
                if dst[x,y]>max:
                    points.append([y-3,x-3])
                    # originImg[x-3,y-3] = [255, 0,0]


        dis=[pow(blkHeight/8,2)+pow(blkWidth/8,2)]*4
        ox=[[int(left*width),int(bottom*height)],[int(right*width),int(bottom*height)],[int(left*width),int(top*height)],[int(right*width),int(top*height)]]
        resPoints=copy.deepcopy(ox)
        for i in range(len(points)):
            for x in range(4):
                tempDis=pow(ox[x][0]-points[i][0],2)+pow(ox[x][1]-points[i][1],2)
                if tempDis<dis[x]:
                    resPoints[x]=points[i]
                    dis[x]=tempDis
        time_e=time.time()
        print "------------conner:",str(time_e-time_s)
        #blackBoard's angle points
        #print resPoints
        # for i in range(len(resPoints)):
        #     img[resPoints[i][1],resPoints[i][0]] = [0, 0, 0]
            #img[ox[i][0],ox[i][1]]=[0,0,0]

        #perspective transform
        time_s=time.time()
        CanvasPoints = np.float32([[0, 0], [blkWidth,0], [0,blkHeight], [blkWidth,blkHeight]])
        PerspectiveMatrix = cv.getPerspectiveTransform(np.array(resPoints,np.float32), np.array(CanvasPoints))
        PerspectiveImg = cv.warpPerspective(originImg, PerspectiveMatrix, (blkWidth, blkHeight))
        time_e=time.time()
        print "---------------pers:",str(time_e-time_s)
        # cv.imshow("orgimg",originImg)
        # cv.imshow("img",img)
        # cv.imshow("res",PerspectiveImg)
        # cv.waitKey(0)
        return PerspectiveImg

def Init():
    global cap,handler
    cap = cv.VideoCapture(cc.CAMERA_NUM)
    cap.read()

    detector = todo()
    handler = handlerObj(detector, rmNoise_type=1)
def ReleaseCam():
    global cap
    cap.release()
#detector create a rectangle that contain the blackboard


class picManager:
    recentPic = {}
    blkPic_dict = {}
    pptPic_dict = {}

    def __init__(self, interval):
        self.blkBoard_mutex = threading.Lock()

        self.pptPic_mutex = threading.Lock()

        self.interval = interval

    def uploadPPT_pic(self,reqTime,image):
        nowTime=str(int(time.time()))
        #print "nowtime:"+nowTime
        if reqTime+cc.VALID_PACK > int(nowTime):
            if self.pptPic_mutex.acquire(cc.INTERVAL):
                image.save(pc.PIC_LOCAL["2"]+nowTime+".jpeg")
                self.pptPic_dict[nowTime]=nowTime
                self.recentPic["2"]=nowTime
                self.pptPic_mutex.release()
            return 1
        return 0

    def getBlkboard_pic(self, reqTime):
        image = b""
        if reqTime < self.interval + int(self.recentPic["blkBoard"]):
            '''read file with file mutex
            '''
            if self.blkBoard_mutex.acquire(cc.INTERVAL):
                picTime = self.recentPic["blkBoard"]
                with open(pc.PIC_LOCAL["1"] + picTime + ".jpeg", "rb") as f:
                    image = f.read()
                self.blkBoard_mutex.release()
        else:  # therer is no useful pic,then capture a new pic
            nowTime = str(int(time.time()))
            global cap,handler
            for i in range(4):
                cap.grab()
            res, frame = cap.read()  # capture a new pic
            if res:  # change recentPic's time
                '''write file with file mutex
                '''
                if self.blkBoard_mutex.acquire(cc.INTERVAL):
                    frame=handler.originHandle(frame)
                    cv.imwrite(pc.PIC_LOCAL["1"] + nowTime + ".jpeg", frame)
                    self.recentPic["blkBoard"] = nowTime
                    self.blkPic_dict[nowTime] = nowTime  # add a new pic name to blkPic_list
                    self.blkBoard_mutex.release()
            with open(pc.PIC_LOCAL["1"] + nowTime + ".jpeg", "rb") as f:
                image = f.read()
        return image

    def getPPT_pic(self, reqTime):
        image=b""
        if self.pptPic_mutex.acquire(cc.INTERVAL):
            if int(self.recentPic["ppt"])+pc.DELETE_INTERVAL> reqTime:
                with open(pc.PIC_LOCAL["2"]+self.recentPic["ppt"]+"jpeg","rb") as f:
                    image=f.read()
            self.pptPic_mutex.release()
        return image

    def DeleteUseless_pic(self):
        if self.blkBoard_mutex.acquire(cc.INTERVAL):
            for key in self.blkPic_dict.keys():
                value = self.blkPic_dict[key]
                nowTime = int(time.time())
                if int(value) + pc.DELETE_INTERVAL < nowTime:
                    os.remove(pc.PIC_LOCAL["1"] + value + ".jpeg")
                    print "remove",value
                    self.blkPic_dict.pop(key)
            self.blkBoard_mutex.release()
        if self.pptPic_mutex.acquire(cc.INTERVAL):
            for key in self.pptPic_dict.keys():
                nowTime = int(time.time())
                value = self.pptPic_dict[key]
                if int(value) + pc.DELETE_INTERVAL < nowTime:
                    os.remove(pc.PIC_LOCAL["1"] + value + ".jpeg")
                    self.pptPic_dict.pop(key)
            self.pptPic_mutex.release()

'''
regist pic type to picManager
'''

def registerPic_type(type):
    picManager.recentPic[type] = "0"


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
