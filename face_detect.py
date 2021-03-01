import cv2
import numpy as np
import torch
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


# 用于测试集：返回confidence最大的人脸框左上角与右下角的坐标
def get_test_box(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.UMat(cv2.resize(img, (300, 300))), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw faces on image
    ans=[]
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                if len(ans)==0 or ans[2]<confidence:
                        ans=[(x,y),(x1,y1),confidence]
    return ans[0],ans[1]

# 用于训练集
def get_train_box(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.UMat(cv2.resize(img, (300, 300))), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw faces on image
    ans=[]
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                ans.append(((x,y),(x1,y1)))
    return ans
# 用于训练过程，利用已知的label点坐标来反推应该返回哪一个人脸框
def get_face(img,label):
    faces=get_train_box(img)
    for ((x,y),(x1,y1)) in faces:
        if label[29][0]>x and label[29][1]>y and label[29][0]<x1 and label[29][1]<y1:
            return (x,y),(x1,y1)
    return []