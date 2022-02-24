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


path1 = "D:\\basketball\\basketball"
path1s = "D:\\differenceb.png"
path2 = "D:\\dumptruck\\dumptruck"

dt1 = cv2.imread(path2+"1.bmp")
dt2 = cv2.imread(path2+"2.bmp")

bb1=cv2.imread(path1+"1.bmp")
bb2=cv2.imread(path1+"2.bmp")
bb=cv2.imread(path2)

gbb1=cv2.cvtColor(dt1, cv2.COLOR_BGR2GRAY)
gbb2=cv2.cvtColor(dt2, cv2.COLOR_BGR2GRAY)

#---------Backgound Remove---------------------------------------------------------------------
kernel = np.ones((3,3), np.uint8)
ker=np.array([[0,1,0],
              [1,1,1],
              [0,1,0]],dtype=np.uint8)
itr=1


difference = cv2.absdiff(gbb1,gbb2)
_ ,diff = cv2.threshold(difference,6,255,cv2.THRESH_BINARY)

#KNN = cv2.createBackgroundSubtractorKNN()
#fgmask= KNN.apply(bb)

#fgmask = cv2.erode(fgmask,kernel,itr)
#fgmask = cv2.dilate(fgmask,kernel,itr)

#cv2.rectangle(gbb1,(10,2),(100,20,),(255,0,0),-1)
#fgmask[np.abs(fgmask)<250]=0


#--------------------------------------------------------------------------------------

hsv1 = np.zeros_like(gbb1)
hsv1[...,1] = 255


flow=cv2.calcOpticalFlowFarneback(gbb1, gbb2, None, 0.5,3,15,3,5,1.2,0)

h,w=gbb1.shape[:2]
step = 9
y,x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
fx, fy = flow[y, x].T
lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
lines = np.int32(lines)

#------------------以下向量場的線尚需調整
line = []
startpoint=[]
endpoint=[]
u = []
v = []
X = []
Y = []
diffP = []
vecP = []
#------------------------------------------------
width = len(diff[0]) #359
height = len(diff[1]) #269

for i in range(269):
    for j in range(359):
        if diff[i][j] == 255:
            diffP.append(tuple(np.array([i,j])))
    
for l in lines:
    if l[0][0]-l[1][0]>0.01 or l[0][1]-l[1][1]>0.01:
        line.append(l)
        pos = np.array([l[0][1],l[0][0]])
        position = tuple(pos)
        T = position in diffP
        if T == True:
            print(position)
            u.append((l[1][0]-l[0][0])/2)  #X到X
            v.append((0-(l[1][1]-l[0][1]))/2)  
            X.append(l[0][0])
            Y.append(l[0][1])

print(type(diffP))
        

     
        #v.append(np.array([l[0][1],l[1][1]]))    #Y到Y
RY=list(reversed(Y))

for l in range(len(RY)):
    RY[l]=RY[l]-60

print(len(u))
print(len(v))

VectFd=cv2.polylines(bb1, line, 0, (0,255,0))



#cv2.arrowedLine(bb1,startpoint[280],endpoint[280],(0,255,0) )

#X = np.linspace(0,360,912)
#Y = np.linspace(0,270,912)
#plt.figure(dpi=300)
plt.figure(dpi=300)
plt.quiver(X,RY,u,v)
plt.xticks(np.arange(0,360,30))
plt.yticks(np.arange(0,300,30))

plt.gca().set_aspect("equal")
plt.show()

#-------------------------------------------------------------------------------------
#print(type(startpoint[12])
print("-----------------")


cv2.namedWindow("Basketball1",cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow("KNN",cv2.WINDOW_AUTOSIZE)
cv2.imshow("Basketball1",diff) #VectFd
#cv2.imshow("diff",diff)
#cv2.imshow("bbb2",bbb2)

click= cv2.waitKey(0)
if click == ord("s"):
    cv2.imwrite(path1s,diff)#VectFd
    print("The vector field image have been saved successfully.")

cv2.destroyAllWindows()
