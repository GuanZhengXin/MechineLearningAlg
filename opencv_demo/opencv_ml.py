import numpy as np
import cv2 as cv

# opencv的机器学习
# 样本+特征+分类器 --> face
# haar  hog

#视频分解提取图片
# vidio = cv.VideoCapture('./test/1.mp4')
# isOpened = vidio.isOpened()
# i = 0
# while(isOpened):
#     if i==10:
#         break
#     else:
#         i = i+1
#     fileName = 'shizhi'+str(i)+'.jpg'
#     print(fileName)
#     (isSucess,frame) = vidio.read()
#     if isSucess is True:
#         cv.imwrite('./test/'+fileName,frame,[cv.IMWRITE_JPEG_QUALITY,100])
# print('end')

# haar+adaboost 人脸识别
# img= cv.imread('./test/renlian.jpg',1)
# cv.imshow('src',img)
# eye_xml = cv.CascadeClassifier('haarcascade_eye.xml')
# face_xml = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# faces = face_xml.detectMultiScale(gray,1.1,10)
# print('faces=',len(faces))
# for (x,y,w,h) in faces:
#     cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     gray_eye = gray[x:x+w,y:y+h]
#     color_eye = img[x:x+w,y:y+h]
#     eyes = eye_xml.detectMultiScale(gray_eye,1.1,5)
#     print('eyes=',len(eyes))
#     for (i,j,m,n) in eyes:
#         cv.rectangle(color_eye,(i,j),(i+m,j+n),(0,0,255),1)
# cv.imshow('dest',img)
# cv.waitKey(0)


