import torch
from shufflenetv2 import shufflenetv2
import argparse
from data import ImageDataTrain,get_loader
import torch.nn as nn
from face_detect import get_face,get_train_box,get_test_box
import cv2
import numpy as np
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1) # only support 1 now

    # Train data
    config = parser.parse_args()

    train_dataset=ImageDataTrain()
    train_loader=get_loader(train_dataset,config)

    model=shufflenetv2()
    if(config.cuda):
        model=model.cuda()
    # print('Network Structure: ')
    # with open("model_2_binary.txt","w") as f:
    #     print(model,file=f)

    opt_adam=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    loss_func=nn.MSELoss()

    all_loss={}

    for epoch in range(config.epoch):
        print('epoch: ',epoch)
        for step,sample in enumerate(train_loader):

            sal_image,sal_label=sample['sal_image'],sample['sal_label']


            # face=np.array(torch.squeeze(sal_image).transpose((2, 0, 1)))
            # cv2.imshow('img',face)
            # cv2.waitKey(10)

            if config.cuda:
                sal_image,sal_label=sal_image.cuda(),sal_label.cuda()

            pre=torch.squeeze(model(sal_image))
            output=torch.squeeze(sal_label)

            loss=loss_func(pre,output)
            opt_adam.zero_grad()
            loss.backward()
            opt_adam.step()
            print('step: ', step," loss: ",loss)


    print(all_loss)



