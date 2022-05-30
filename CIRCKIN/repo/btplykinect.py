#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import numpy as np
import os
import time

window_depth_name = 'Depth'
PEROLD=0
perimeter=0
NEV=1;
ACT=2;
X=0
Y=0
XO=X
YO=Y

threshold = 95
current_depth = 267
current_area=2500
g1=31
g2=36
pix=150
m=0


def change_pix(value):
    global pix
    pix = value


def change_g2(value):
    global g2
    g2 = value

def change_g1(value):
    global g1
    g1 = value

def change_threshold(value):
    global threshold
    threshold = value

def change_depth(value):
    global current_depth
    current_depth = value

def change_area(value):
    global current_area
    current_area = value

def show_depth2():
    global threshold;#,w,p;
    global current_depth,FIM,quad,ACT;
    global X,Y,XO,YO;
    global current_area,PEROLD,g1,g2,pix,perimeter
    g1x=g1/100
    g2x=g2/100
    dtip=10
    n=0;
    global m;
    th=threshold/100;
    s=''
    kernelSize = (7, 7)
    # loop over the kernels sizes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)

    depthx, timestamp = freenect.sync_get_depth()


    #depthx = 1.0 * np.logical_and(depthx >= (current_dep/th) - threshold,
    #                        depthx <= (current_depth) + threshold)
    depthx = np.reciprocal(1.0*depthx)
    mx=max(np.ravel(depthx))
    depthx=(depthx>(th*mx)).astype(np.uint8)
    ####################################
    depthx = np.fliplr(depthx)
    #deptip = np.fliplr(deptip)
    depthx = depthx.astype(np.uint8)
    #deptip = deptip.astype(np.uint8)
    W=depthx.shape[1];
    H=depthx.shape[0];
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(depthx,127,255,0)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(depthx, connectivity, cv2.CV_32S)
    #outip = cv2.connectedComponentsWithStats(deptip, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    ################
    #print(numLabels)
    areas=stats[:,4]
    nr=np.arange(numLabels)
    ranked=sorted(zip(areas,nr))
    z1=ranked[:-1]
    N1=len(z1)
    depthx= np.zeros((labels.shape))
    drawing = np.zeros((depthx.shape[0], depthx.shape[1], 3), dtype=np.uint8)

    print('*********************')
    #print(stats[1:,4])
    #print(numLabels)
    Amax=max(stats[1:,4])
    #print("AM:",Amax)
    #lb= np.where(stats[1:,4]==Amax)
    #print(Amax,lb)#,lb+1,stats[lb+1,4])
    for j in range(1,numLabels):
        if stats[j][4]==Amax:
            #print(j,stats[j][4])
            depthx[labels==j]=255
            drawing[labels==j]=255
            closing = cv2.morphologyEx(depthx, cv2.MORPH_OPEN, kernel)
            retx,threshx = cv2.threshold(closing,127,255,0)
            drawing = np.zeros((depthx.shape[0], depthx.shape[1], 3), dtype=np.uint8)
            drawing[threshx==255]=255
            drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
            output2 = cv2.connectedComponentsWithStats(drawing.astype(np.uint8), connectivity, cv2.CV_32S)
            (numLabelsx, labelsx, statsx, centroidsx) = output2
            if numLabelsx>1:
                Amaxx=max(statsx[1:,4])
                for k in range(1,numLabelsx):
                    if statsx[k][4]==Amaxx:
                        drawing = np.zeros((depthx.shape[0], depthx.shape[1], 3), dtype=np.uint8)
                        drawingx= np.zeros((depthx.shape[0], depthx.shape[1], 3), dtype=np.uint8)
                        drawing[closing==255]=255
                        drawingx[closing==255]=255
                        #drawing=closing
                        #drawingx=closing
                        cv2.circle(drawing, (int(centroidsx[k][0]),int(centroidsx[k][1])), 50, (0, 255, 0),1,8,0)
                        p1x=statsx[k][0];p1y=statsx[k][1];
                        p2x=p1x+statsx[k][2];p2y=p1y+statsx[k][3];
                        cv2.rectangle(drawing,(int(p1x),int(p1y)),(int(p2x),int(p2y)),(255,255,0),3)

                        img_gray = cv2.cvtColor(drawingx,cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(img_gray, 127, 255,0)
                        contours, hierarchy = cv2.findContours(thresh,2,1)
                        cnt = contours[0]
                        hull=cv2.convexHull(cnt)
                        per=cv2.arcLength(hull,True)
                        ar=cv2.contourArea(hull,True)
                        cir=(4*np.pi*Amaxx)/(per**2)
                        cvnx=Amaxx/ar
                        wx=" Wbox:"+str("{:.2f}".format(statsx[k][2]))+" "
                        hx=wx+"Hboc:"+str("{:.2f}".format(statsx[k][3]))+" "
                        R=hx+" R:"+str("{:.2f}".format( float(statsx[k][2]) / float(statsx[k][3]) ))
                        print(per,ar,cir,cvnx)
                        cv2.drawContours(drawing,cnt,-1, (255,0,0),3)
                        cv2.drawContours(drawing,[hull],-1, (255,255,255),2)
                        st='CIR:'+str("{:.2f}".format(cir))+"-"+'CONV:'+str("{:.2f}".format(cvnx))
                        cv2.putText(drawing,str(st)+R,(10,H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
                        print(len(statsx[k]))
                        #if len(statsx[k])==5:
                        #    ACT=statsx[k][2]
                        #cv2.putText(drawing,str(ACT),(100,H-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
                        #cv2.putText(drawing,str(ACT),(100,H-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
                        break
            break
    print('*********************')

    X=(1/(1-2*g1x))*float(centroids[1][0]/W)-(g1x/(1-2*g1x));
    Y=(1/(1-2*g2x))*float(centroids[1][1]/H)-(g2x/(1-2*g2x));
    #        #print(X,Y,g1x,g2x)
    if X>=1.0:
        X=1.0
    if Y>=1.0:
        Y=1.0
    if X<=0.0:
        X=0.0
    if Y<=0.0:
        Y=1.0
    if X>0 and Y>0 and X<1 and Y<1:
        s=str("{:.2f}".format(X))+" "+str("{:.2f}".format(1-Y))+" "+str(ACT);
    cv2.rectangle(drawing,(int(g1x*W),int(g2x*H)),(int((1-g1x)*W),int((1-g2x)*H)),(0,255,0),3)
    #time.sleep(0.1)
    #n+=1
    #m+=1
    cv2.imshow('Depth', drawing)
    return s


cv2.namedWindow('Depth')
#cv2.namedWindow('CMDepth')
#cv2.namedWindow('Video')
cv2.createTrackbar('threshold', 'Depth', threshold,     100,  change_threshold)
#cv2.createTrackbar('depth',     'Depth', current_depth, 2048, change_depth)
#cv2.createTrackbar('minarea',     'Depth', current_area, 5500, change_area)
cv2.createTrackbar('g1', 'Depth', g1, 40, change_g1)
cv2.createTrackbar('g2', 'Depth', g2, 40, change_g2)
#cv2.createTrackbar('pix', 'Depth', pix, 200, change_pix)




print('Press ESC in window to stop')
FIM=0;
p='fifo'
#w=open(p, 'w')
j=0
while 1:

        #w=open(p, 'w')
        a=show_depth2()
        #if len(a)!=0:
        #    #print(a)
        #    w=open(p, 'w')
        #    w.write(a+'\n')
        #    w.flush()
        #    w.close()
        j+=1
        if cv2.waitKey(10) == 27:
            freenect.sync_stop()
            cv2.destroyAllWindows()
            break
