#!usr/bin/python
#coding=utf-8

import time

from bottle import HTTPResponse,request,app,route,response
from util import picManager as pm
from config import cameraConfig as cc

@route("/upload",method="post")
def uploadPic():
    print "--------upload,header--------"
    for key in request.headers:
        print key+":"+request.headers[key]
    print "---------------------------"
    print "--------upload,body--------"
    print request.body.read()
    print "---------------------------"

    try:
        requestTime=request.forms["time"]
        #type=request.forms["type"]
    except Exception as err:
        print "--------upload,froms error--------"
        print err
        print "----------------------------------"
        body = "lack of time or type"
        return HTTPResponse(status=404, body=body)
    # para error,include null and validation
    if  not requestTime:
        body="type or time is null"
        return HTTPResponse(status=401,body=body)
    '''
    if int(type)>cc.PIC_TYPENUM:
        body="type error"
        return HTTPResponse(status=402,body=body)
    '''

    try:
        image = request.files.get("image")
        res = pm.picManager.uploadPPT_pic(int(requestTime), image)
    except Exception as err:
        print "--------upload,picManager error--------"
        print err
        print "----------------------------------"
        body = "read picture error"
        return HTTPResponse(status=501, body=body)

    if res==0:
        body="over time package"
        return HTTPResponse(status=403, body=body)

    else:
        body="upload success"
        return HTTPResponse(status=200,body=body)
