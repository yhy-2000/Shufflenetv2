import torch
from torch.utils import data
import cv2
import numpy as np
import os
from face_detect import get_face,get_test_box,get_train_box


class ImageDataTrain(data.Dataset):
    def __init__(self, path='./300W/01_Indoor/'):
        self.path = path
        with open(path + "01_train_list.txt", "r") as f:
            self.img_list = [x.strip() for x in f.readlines()]
            self.label_list = [x.replace(".png", ".pts") for x in self.img_list]

    # 返回只包含人脸的tensor，以及对应的归一化关键点坐标
    def __getitem__(self, idx):
        photo_name = self.img_list[idx]
        label_name = self.label_list[idx]
        label = read_pts(self.path + label_name)

        (x,y),(x1,y1),img = load_image(self.path + photo_name,label)

        img = torch.Tensor(img)

        # 关键点坐标要归一化处理，便于收敛
        label[:,0] = (label[:,0]-y) / (y1-y)
        label[:,1] = (label[:,1]-x) / (x1-x)

        label = label.reshape(-1, 1)
        label = torch.Tensor(label)

        sample = {'img_name':photo_name, 'sal_image': img, 'sal_label': label,'h':x1-x,'w':y1-y}

        return sample

    def __len__(self):
        return len(self.img_list)


def read_pts(path):
    file = np.loadtxt(path, comments=("version:", "n_points:", "{", "}"))
    return file


def load_image(path,label):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    img = cv2.imread(path)

    # 如果可以检测到人脸，则进行裁剪
    if(len(get_face(img,label))):
        (x,y),(x1,y1)=get_face(img,label) #返回人脸框的位置
        img=img[y:y1,x:x1]
    # 否则进行缩放
    else:
        x,y=img.shape[0:2]
        img=cv2.resize(img,(x//5,y//5))


    # 将图像转换成[N,C,W,H]格式的tensor
    in_ = np.array(img, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return (x,y),(x1,y1),in_


# train_loader
def get_loader(dataset, config):
    shuffle = True
    data_loader = data.DataLoader(dataset=dataset, batch_size=
    config.batch_size, shuffle=shuffle, pin_memory=True)
    return data_loader

# li=ImageDataTrain()
