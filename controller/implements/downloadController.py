#!usr/bin/python
#coding=utf-8

import time

from bottle import HTTPResponse,request,app,route,response
from util import picReader as pr
from config import cameraConfig as cc

@route("/download",method="get")
def downloadPic():

    type=request.query.type;
    requestTime=request.query.time;

    if not type or not requestTime:
        body="code:401\nreason:type or time is null"
        return HTTPResponse(status=400,body=body)

    nowTime=time.time()
    if float(requestTime)+cc.VALID_PACK <float(nowTime):
        body="code:402\nreason:over time,it is not valid"
        return HTTPResponse(status=400, body=body)

    res=pr.downloadPic(type,nowTime)
    if not res:
        body="code:501\nreason:server error"
        return HTTPResponse(status=500, body=body)

    response.set_header("content_type","image/jpeg")
    return  res
    #return response