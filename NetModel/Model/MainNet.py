from Model.DDT import DDT_module
import torch
from torch import mode, nn
import torchvision.models as models
import sys
import os

sys.path.append(os.getcwd())

# from utils.config import pth_path
# from DDT import DDT_module


class MainNet(nn.Module):
    def __init__(self, num_class, channels=512):
        super(MainNet, self).__init__()
        self.backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.predict = nn.Linear(channels,num_class)
        self.ddt = DDT_module(model=self.backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x,epoch,batch_idx,status='test'):
        origin_x  = x
        # 截取放大后的图像
        # x,bboxes_list = self.ddt(x)
        # # batch channel h w
        # obj_img = x 
        # x = torch.from_numpy(x).cuda()
        # 卷积提取特征分类
        x = self.backbone(origin_x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.predict(x)
        # if status == "train":

        # else :
           
        #return obj_img,bboxes_list,x
        return x

# def getBackbone(backbone_name="vgg19", pretrained=True):
#     # 获取主干网络及其预训练方式
#     base_backbone = eval(backbone_name)(pretrained=pretrained)
#     # 所有网络的子结构
#     all_base_modules = [m for m in base_backbone.children()]

#     backbone = None

#     if backbone_name == "vgg16":
#         feature_extractor = all_base_modules[0]
#         backbone = feature_extractor[0:30]
#     elif backbone_name == "vgg19":
#         feature_extractor = all_base_modules[0]
#         backbone = feature_extractor[0:35]
#     elif backbone_name in ["resnet50", "resnet101"]:
#         layers = all_base_modules[:-3]  # 不包含最后三个子模块
#         backbone = nn.Sequential(*layers)
#     else:
#         print(backbone_name)
#         raise NotImplementedError()

#     return backbone


if __name__ == "__main__":
    import os, sys

    # 当前文件的绝对路径
    root_path = os.path.abspath(__file__)

    model = MainNet()
    input = torch.randint(0, 255, (2, 448, 448, 3), dtype=torch.float)
    output = model(input)
    print(output.shape)

    # backbone = getBackbone()
    # print(backbone)
    