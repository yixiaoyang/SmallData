#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np

'''
@func       根据HoughLines转换出直线点
@param      rho 距离
@param      theta 角度
'''
def rt_to_point(img, rho, theta):
    #垂直直线
	if (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
        #该直线与第一行的交点
		pt1 = (int(rho/np.cos(theta)),0)
		#该直线与最后一行的焦点
		pt2 = (int((rho-img.shape[0]*np.sin(theta))/np.cos(theta)),img.shape[0])
		return pt1, pt2
	else:
        #水平直线,  该直线与第一列的交点
		pt1 = (0,int(rho/np.sin(theta)))
		#该直线与最后一列的交点
		pt2 = (img.shape[1], int((rho-img.shape[1]*np.cos(theta))/np.sin(theta)))
		return pt1, pt2

image = cv2.imread("test8.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = image.copy()

# 二值化，阈值100,255
ret, binary = cv2.threshold(gray, 80,255,cv2.THRESH_BINARY)

# 预处理降噪
#binary = cv2.pyrMeanShiftFiltering(image, 25, 10)

# 中值平滑，消除噪声
binary = cv2.medianBlur(binary,7)
#binary = cv2.medianBlur(binary,5)
#binary = cv2.medianBlur(binary,1)
#binary = cv2.blur(binary,(5,5))

cv2.imwrite("GaussianBlur.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 分析轮廓
_,contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image, contours, -1, (0,0,255),1)

# TODO:获取4个标注点组成RECT
if len(contours) == 0:
    print("detect contour failed")
    exit(0)

# TODO: 角点检测
print("检测角点...")
corners = cv2.cornerHarris(binary,2,3,0.04)
print (corners)

# hough 线检测
edges = cv2.Canny(binary, 0, 60, apertureSize = 3)
cv2.imwrite("canny.png", edges, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 像素1,角度np.pi/180
# lines = cv2.HoughLines(edges, 1, np.pi/180, 118)
# print("hough lines: %d"%(len(lines[0])))
# for line in lines:
#     for l in line:
#         pt1,pt2 = rt_to_point(binary, l[0],l[1])
#         cv2.line(image,pt1,pt2,color=(255,255,0))
# print(lines)
lines = cv2.HoughLinesP(edges, 1, (np.pi*1)/180, threshold=80, minLineLength=70)
if lines:
    if len(lines) != 0:
        print(len(lines))
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,255),3)
print(lines)

# 获取最小矩形包络
rect = cv2.minAreaRect(contours[0])
#print(rect)
box = cv2.boxPoints(rect)
box = np.int0(box)
box = box.reshape((-1,1,2))
#print(box)
cv2.polylines(image,[box],True,(0,255,0))

# Calculates a perspective transform from four pairs of the corresponding points.
#canvas = [[0,0],[]]
#transMat = cv2.getPerspectiveTransform(src=None, dst=None)

cv2.imwrite("contourss.png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
