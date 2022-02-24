# -*- coding: utf-8 -*-
"""
Course:Machine Vision 
Project: Draw the optical flow with two image and improve the algorithm
Author: David Li
Create date:2021.12.21

"""

import cv2
import numpy as np
import time 
import matplotlib.pyplot as plt


path1 = "D:\\1_NSYSU\\1_MasterDegree\MachineVision\\basketball\\"
path2 = "D:\\1_NSYSU\\1_MasterDegree\\MachineVision\\dumptruck\\"
savepath = "D:\\OpticalFlow_VectorField.png"

bkb1=cv2.imread(path1+"basketball1.bmp")
bkb2=cv2.imread(path1+"basketball2.bmp")

dt1 = cv2.imread(path2+"dumptruck1.bmp")
dt2 = cv2.imread(path2+"dumptruck2.bmp")

#__________________image preprocessing____________________
def preprocessing(org1,org2):
    G_img1=cv2.cvtColor(org1, cv2.COLOR_BGR2GRAY)
    G_img2=cv2.cvtColor(org2, cv2.COLOR_BGR2GRAY)
    return org1, G_img1 ,G_img2


#________draw the optical flow on the first frame________
def DrawOpticalFlow(frame1,frame2):
    
    line = []
    startpoint=[]
    endpoint=[]
    u = []
    v = []
    X = []
    Y = []
    
    flow=cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5,3,15,3,5,1.2,0)
    h,w=frame1.shape[:2]
    step = 9
    y,x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, 1200).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    for l in lines:
        if l[0][0]-l[1][0]>0.5 or l[0][1]-l[1][1]>0.5:
            line.append(l)
            u.append((l[1][0]-l[0][0])/2)  #X到X
            v.append((0-(l[1][1]-l[0][1]))/2)
            X.append(l[0][0])
            Y.append(l[0][1])
            #v.append(np.array([l[0][1],l[1][1]]))    #Y到Y
    RY=list(reversed(Y))
    
    return line, u, v, X, RY
    

#_________plot the optical flow on the windows______
def PlotOpticalFlow(org1,line, u, v, X, RY):
    VectFd=cv2.polylines(org1, line, 0, (0,255,0))
    #cv2.arrowedLine(bb1,startpoint[280],endpoint[280],(0,255,0) )
    #X = np.linspace(0,360,912)
    #Y = np.linspace(0,270,912)
    plt.figure(dpi=300)
    plt.quiver(X,RY,u,v)
    plt.xticks(np.arange(0,360,30))
    plt.yticks(np.arange(0,300,30))

    plt.gca().set_aspect("equal")
    plt.show()
    print("-----------------")

    cv2.namedWindow("Basketball1",cv2.WINDOW_AUTOSIZE)

    cv2.imshow("Basketball1",VectFd)

    click= cv2.waitKey(0)
    if click == ord("s"):
        cv2.imwrite(savepath,VectFd)
        print("The vector field image have been saved successfully.")

    cv2.destroyAllWindows()
    
    
    
#______Excute the funtion_______
org1,frame1, frame2 = preprocessing(dt1, dt2)
line, u, v, X, RY = DrawOpticalFlow(frame1,frame2)
PlotOpticalFlow(org1,line, u, v, X, RY)