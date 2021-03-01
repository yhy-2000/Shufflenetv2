import numpy as np
import cv2
import os

def read_pts(path):
    file = np.loadtxt(path, comments=("version:", "n_points:", "{", "}"))
    return file


if __name__=='__main__':
    # for p,d,filenames in os.walk("./300W"):
    #     path=p+"/"
    #     for file in filenames:
    #         if file.endswith(".png"):
    #             img=cv2.imread(path+file)
    #             points=read_pts(path+file.replace(".png",".pts"))
    #             for p in points:
    #                 # cv2.circle(img, (80, 80), 30, (0, 0, 255), -1)
    #                 cv2.circle(img,(int(p[0]),int(p[1])),2,(0,0,255),-1)
    #             cv2.imshow("img",img)
    #             cv2.waitKey(10)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 60), 30, (0, 0, 255))
    cv2.imshow('img', img)
    cv2.waitKey(10000)

