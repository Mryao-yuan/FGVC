from ast import Import
from Model.MainNet import MainNet
import torch
import os

# 测试模型的正确性
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = MainNet(200, 512).cuda()
input = torch.randint(0, 255, (2, 3, 448, 448), dtype=torch.float)
obj, bbox, x = model(input.cuda(), 1, 1)
print(bbox)
print(x.max(1, keepdim=True)[1])
