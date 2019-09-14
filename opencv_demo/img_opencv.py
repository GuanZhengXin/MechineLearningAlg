import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

# 图片的缩放
#resize 原理
#wdith = x*(height/dest_height)
#height = x*(wdith/dest_wdith)
# img =cv.imread('test.jpg',1)
# imginfo = img.shape  #(630,500,3)height width color_weight=3
# print(imginfo)
# height = int(imginfo[0]/2)
# width = int(imginfo[1]/2)
# print(width,height)
# newimg = cv.resize(img,dsize=(width,height))
# cv.imshow('newimg',newimg)
# cv.waitKey(0)

#图片的剪切
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# dst = img[100:200,100:300]
# imgnew = cv.imshow('dest',dst)
# cv.waitKey(0)

#图片的位移
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# dst = np.zeros(shape=(imginfo),dtype=np.uint8)
# for i in range(0,height-100):
#     for j in range(0,width-50):
#         dst[i+100,j+50] = img[i,j]
# cv.imshow('dst',dst)
# cv.waitKey(0)

# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# matshift = np.float32([[1,0,50],[0,1,150]])
# dst = cv.warpAffine(img,matshift,(height,width))
# cv.imshow('dst',dst)
# cv.waitKey(0)

#仿射变换
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# matsrc = np.float32([[0,0],[0,height],[width,0]]) # 位置对于变换 左上 左下角 右上角
# matdest = np.float32([[50,50],[200,height-200],[width-50,20]])
# matmix = cv.getAffineTransform(matsrc,matdest)
# dest = cv.warpAffine(img,matmix,(height,width))
# cv.imshow('dest',dest)
# cv.waitKey(0)

#图片旋转
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# matRotate = cv.getRotationMatrix2D((height*0.5,width*0.5),45,0.5)
# dest = cv.warpAffine(img,matRotate,(height,width))
# cv.imshow('dest',dest)
# cv.waitKey(0)

#灰度计算
# cv.imread('',0)
# gray = (B+G+R)/3
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# dest = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('dest',dest)
# cv.waitKey(0)

#灰白颜色反转 255-gray
# img = cv.imread('test.jpg',0)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# dest = np.zeros((height,width,3),np.uint8)
# for i in range(height):
#     for j in range(width):
#         gray = img[i,j]
#         dest[i,j] = 255-gray
# cv.imshow('dest',dest)
# cv.waitKey(0)

#彩色颜色反转 (255-B,255-G,255-R)
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# dest = np.zeros((height,width,3),np.uint8)
# for i in range(height):
#     for j in range(width):
#         (b,g,r) = img[i,j]
#         dest[i,j] = (255-b,255-g,255-r)
# cv.imshow('dest',dest)
# cv.waitKey(0)

#马赛克
# 矩形框里取一个一样的颜色
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# for m in range(0,height):
#     for n in range(0,width):
#         if m%10==0 and n%10==0:
#             for i in range(0,10):
#                 for j in range(0,10):
#                     img[i+m,j+n] = img[m,n]
# cv.imshow('dest',img)
# cv.waitKey(0)

#毛玻璃 单位矩阵里颜色随机摆放
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# random_shape = 8
# dest = np.zeros((height,width,3),np.uint8)
# for m in range(0,height-random_shape):
#     for n in range(0,width-random_shape):
#         index = int(random.random()*random_shape)
#         dest[m,n] = img[m+index,n+index]
# cv.imshow('dest',dest)
# cv.waitKey(0)


#融合图片
# img1 = cv.imread('test.jpg',1)
# img2 = cv.imread('test1.jpg',1)
# imginfo = img1.shape
# height = int(imginfo[0]/2)
# width = int(imginfo[1]/2)
# img1_pratial = img1[0:height,0:width]
# img2_pratial = img2[0:height,0:width]
# #dest = np.zeros((height,width,3),np.uint8)
# dest = cv.addWeighted(img1_pratial,0.5,img2_pratial,0.5,0)
# cv.imshow('dest',dest)
# cv.waitKey(0)

#边缘检测
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# gary = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# imgnew = cv.GaussianBlur(img,(3,3),0) #滤波
# dest = cv.Canny(imgnew,30,30)
# cv.imshow('dest',dest)
# cv.waitKey(0)

#浮雕
# new_p = grapP0-grayP1+150 边检值相减+150(灰度效果值)
# img = cv.imread('test.jpg',1)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# dest = np.zeros((height,width,3),np.uint8)
# for i in range(0,height):
#     for j in range(0,width-1):
#         grapP0 = int(gray[i,j])
#         grayP1 = int(gray[i,j+1])
#         new_p = grapP0-grayP1+150
#         if new_p>255:
#             new_p = 255
#         if new_p<0:
#             new_p = 0
#         dest[i,j] = new_p
# cv.imshow('dest',dest)
# cv.waitKey(0)

#色调调正
# img = cv.imread('test.jpg',1)
# cv.imshow('src',img)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# dest = np.zeros((height,width,3),np.uint8)
# for i in range(0,height):
#     for j in range(0,width):
#         (b,g,r) = img[i,j]
#         r = r*1.5
#         g = g*1.2
#         if r>255:
#             r = 255
#         if g >255:
#             g = 255
#         dest[i,j] = (b,g,r)
# cv.imshow('dest',dest)
# cv.waitKey(0)

#
# dest = np.zeros((500,500,3),np.uint8)
# cv.ellipse(dest,(250,250),(100,100),0,0,270,(0,0,255))
# cv.imshow('dest',dest)
# cv.waitKey(0)

#直方图均衡化 调整对比度
# 累计灰度概率
# equalizeHist()
# img = cv.imread('test.jpg',1)
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('src',gray)
# dest = cv.equalizeHist(gray)
# cv.imshow('dest',dest)
# cv.waitKey(0)

#彩色均衡化
# img = cv.imread('test.jpg',1)
# cv.imshow('src',img)
# (b,g,r) = cv.split(img) # 分解三个通道
# bH = cv.equalizeHist(b)
# gH = cv.equalizeHist(g)
# rH = cv.equalizeHist(r)
# dest = cv.merge([bH,gH,rH])
# cv.imshow('dest',dest)
# cv.waitKey(0)

#转为YUV 进行均衡化  3通道（channel）
#YUV 明亮对 色度、浓度
# img = cv.imread('test.jpg',1)
# cv.imshow('src',img)
# YUV = cv.cvtColor(img,cv.COLOR_BGR2YUV)
# channel_YUV = cv.split(YUV)
# channel_YUV[0] = cv.equalizeHist(channel_YUV[0])
# new_YUV = cv.merge(channel_YUV)
# dest = cv.cvtColor(new_YUV,cv.COLOR_YUV2BGR)
# cv.imshow('dest',dest)
# cv.waitKey(0)

#灰度直方图 描述灰度等级上出现的概率
# img = cv.imread('test1.jpg',1)
# imgInfo = img.shape
# height = imgInfo[0]
# width = imgInfo[1]
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# count = np.zeros(256,np.float)
# for i in range(255):
#     for j in range(255):
#          level = int(gray[i,j])
#          count[level] = count[level]+1
# for i in range(256):
#     count[i] = count[i]/(height*width)
# x = np.linspace(0,255,256)
# y = count
# plt.bar(x,y,1,alpha=1,color='b')
# plt.show()
# cv.waitKey(0)

#灰度直方图均衡化 源码 累计
# 得到一个映射表 灰度等级=255*累计概率 lever-->255*sum
# img = cv.imread('test.jpg',1)
# imgInfo = img.shape
# height = imgInfo[0]
# width = imgInfo[1]
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# img = cv.imshow('gray',gray)
# count = np.zeros(256,np.float)
# for i in range(height):
#     for j in range(width):
#          gray_level = int(gray[i,j])
#          level = int(gray_level)
#          count[level] = count[level]+1
# for i in range(256):
#     count[i] = count[i]/(height*width)
# #计算累计概率
# sum = float(0)
# for i in range(256):
#     sum = sum + count[i]
#     count[i] = sum
# map = np.zeros(256,np.uint8)
# for i in range(256):
#     map[i] = np.uint8(255*count[i])
# for i in range(height):
#     for j in range(width):
#         level = gray[i, j]
#         gray[i,j] = map[level]
# cv.imshow('dest',gray)
# cv.waitKey(0)

#彩色直方图 描述彩色通道等级上出现的概率
# img = cv.imread('test2.jpg',1)
# imgInfo = img.shape
# height = imgInfo[0]
# width = imgInfo[1]
# count_r = np.zeros(256,np.float)
# count_g = np.zeros(256,np.float)
# count_b = np.zeros(256,np.float)
# for i in range(height):
#     for j in range(width):
#         (b,g,r) = img[i,j]
#         b = int(b)
#         g = int(g)
#         r = int(r)
#         count_b[b] = count_b[b] + 1
#         count_g[g] = count_g[g] + 1
#         count_r[r] = count_r[r] + 1
# for i in range(256):
#     count_b[i] = count_b[i] / (height * width)
#     count_g[i] = count_g[i] / (height * width)
#     count_r[i] = count_r[i] / (height * width)
# x = np.linspace(0,255,256)
# y1 = count_b
# y2 = count_g
# y3 = count_r
# plt.subplot(221)
# plt.bar(x,y1,1,alpha=1,color='b',label='bhist')
# plt.subplot(222)
# plt.bar(x,y2,1,alpha=1,color='g',label='ghist')
# plt.subplot(223)
# plt.bar(x,y3,1,alpha=1,color='r',label='rhist')
# plt.show()
# cv.waitKey(0)

#彩色直方图均衡化 源码
# img = cv.imread('test1.jpg',1)
# cv.imshow('src',img)
# imgInfo = img.shape
# height = imgInfo[0]
# width = imgInfo[1]
# count_b = np.zeros(256,np.float)
# count_g = np.zeros(256,np.float)
# count_r = np.zeros(256,np.float)
# for i in range(height):
#     for j in range(width):
#         (b,g,r) = img[i,j]
#         b_level = int(b)
#         g_level = int(g)
#         r_level = int(r)
#         count_b[b_level] = count_b[b_level] + 1
#         count_g[g_level] = count_g[g_level] + 1
#         count_r[r_level] = count_r[r_level] + 1
# for i in range(256):
#     count_b[i] = count_b[i] / (height * width)
#     count_g[i] = count_g[i] / (height * width)
#     count_r[i] = count_r[i] / (height * width)
# #计算累计概率
# sum_b = float(0)
# sum_g = float(0)
# sum_r = float(0)
# for i in range(256):
#     sum_b = sum_b + count_b[i]
#     count_b[i] = sum_b
#     sum_g = sum_g + count_g[i]
#     count_g[i] = sum_g
#     sum_r = sum_r+ count_r[i]
#     count_r[i] = sum_r
# map_b = np.zeros(256,np.uint8)
# map_g = np.zeros(256,np.uint8)
# map_r= np.zeros(256,np.uint8)
# for i in range(256):
#     map_b[i] = 255 * count_b[i]
#     map_g[i] = 255 * count_g[i]
#     map_r[i] = 255 * count_r[i]
# for i in range(height):
#     for j in range(width):
#         (b,g,r) = img[i, j]
#         b_level = int(b)
#         g_level = int(g)
#         r_level = int(r)
#         b_level = np.uint8(map_b[b_level])
#         g_level = np.uint8(map_g[g_level])
#         r_level = np.uint8(map_r[r_level])
#         img[i,j] = (b_level,g_level,r_level)
# cv.imshow('dest',img)
# cv.waitKey(0)

#亮度增强 rgb+特定的值

#磨皮美白 双边滤波
# img = cv.imread('mopi.jpg',1)
# cv.imshow('src',img)
# dest = cv.bilateralFilter(img,15,150,50)
# cv.imshow('dest',dest)
# cv.waitKey(0)

#高斯滤波
# img = cv.imread('mopi.jpg',1)
# cv.imshow('src',img)
# #高斯内核大小，其中ksize.width和ksize.height可以不同，但是必须为正数和奇数，也可为零，均有sigma计算而来
# #表示高斯函数在X方向的标准偏差
# dest = cv.GaussianBlur(img,(5,5),2)
# dest = cv.blur(img,(5,5),2)  #均值滤波
# dest = cv.medianBlur(img,(5,5)) #中值滤波
# cv.imshow('dest',dest)
# cv.waitKey(0)








