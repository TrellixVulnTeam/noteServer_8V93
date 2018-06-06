#!usr/bin/python
#coding=utf-8

import time

from bottle import HTTPResponse,request,app,route,response
from util import picManager as pm
from config import cameraConfig as cc

@route("/upload",method="post")
def uploadPic():
    for key in request.headers:
        print key+":"+request.headers[key]
    requestTime=request.forms["time"]
    print request.body.read()
    #type=request.forms["type"]
    #para error,include null and validation
    if  not requestTime:
        body="type or time is null"
        return HTTPResponse(status=401,body=body)
    '''
    if int(type)>cc.PIC_TYPENUM:
        body="type error"
        return HTTPResponse(status=402,body=body)
    '''
    image=request.files.get("image")
    res=pm.picManager.uploadPPT_pic(int(requestTime),image)

    if res==0:
        body="over time package"
        return HTTPResponse(status=403, body=body)
    else:
        body="upload success"
        return HTTPResponse(status=200,body=body)
