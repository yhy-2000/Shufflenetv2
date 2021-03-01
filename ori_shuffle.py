import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
model.eval()
with open("model_ori.txt","w") as f:
    print(model,file=f)