#!usr/bin/python
#coding=utf-8

import bottle
import os
import threading
import time
from util import picManager as pr
from config import controllerConfig
from config import serverConfig as sc

#from implements import controllers batch
#dir:implements dir
def registerController(dir=controllerConfig.IMPLEMENTS_DIR):
    #whether it has and it is a dir
    if os.path.exists(dir):
        if os.path.isdir(dir):
            #iterate the dir
            impList=os.listdir(dir)
            dir=dir.replace("/", ".")
            for value in impList:
                #if is a controller file then import

                if value[:2] != "__" and value[-3:]==".py":
                    value=value[:len(value)-3]
                    importStr="from "+dir[2:]+" import "+value
                    exec(importStr,globals())


registerController()
threading._start_new_thread(pr.TD,())
bottle.run(port=sc.PORT,host=sc.HOST)


