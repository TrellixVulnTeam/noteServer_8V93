#!usr/bin/python
#coding=utf-8

import time

from bottle import HTTPResponse,request,route,response
from util import picManager as pm
from config import cameraConfig as cc

@route("/download",method="get")
def downloadPic():

    type=request.query.type;
    requestTime=request.query.time;

    #para error,include null and validation
    if not type or not requestTime:
        body="type or time is null"
        return HTTPResponse(status=401,body=body)

    if int(type)>cc.PIC_TYPENUM:
        body="type error"
        return HTTPResponse(status=402,body=body)

    #package error,include time limit and cookie
    nowTime=time.time()
    if float(requestTime)+cc.VALID_PACK <nowTime:
        body="over time package"
        return HTTPResponse(status=403, body=body)

    #server error
    res=pm.downloadPic(type,int(requestTime))
    if not res:
        body="server error"
        return HTTPResponse(status=501, body=body)


    response.set_header("content_type","image/jpeg")
    return  res
    #return response