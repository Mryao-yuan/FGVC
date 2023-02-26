import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 使用的是 vgg19的架构
from .vgg import vgg19_bn
# test
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RACNN(nn.Module):
    # input [2, 3, 448, 448]
    def __init__(self, num_classes, pretrained=True):
        super(RACNN, self).__init__()
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = False  
        # 三个不同scale下的 特征提取
        self.b1 = vgg19_bn(num_classes=1000, pretrained='imagenet')
        self.b2 = vgg19_bn(num_classes=1000, pretrained='imagenet')
        self.b3 = vgg19_bn(num_classes=1000, pretrained='imagenet')
        # 特征池化
        self.feature_pool1 = nn.AvgPool2d(kernel_size=28, stride=28)
        self.feature_pool2 = nn.AvgPool2d(kernel_size=14, stride=14)

        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 堆叠两个全连接层,输出的是受关注的区域，两个apn不共享参数
        self.apn1 = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),     # 类似通道注意力，输出三个值
            nn.Sigmoid(),
        )
        # 输出t_x,t_y,t_l,形状
        self.apn2 = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        # 特征区域的裁剪
        self.crop_resize = AttentionCropLayer()
        # 三个分类器 c1,c2,c3
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.classifier3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 通过 conv5_4 响应区最高的地方来选择区域
        # torch.Size([batch_size, 512, 28, 28])
        conv5_4 = self.b1.features[:-1](x)  # 不使用最后一个全连接层,输入x(特征)
        pool5 = self.feature_pool1(conv5_4) # 特征池化
        atten1 = self.apn1(self.atten_pool(conv5_4).view(-1, 512 * 14 * 14)) # 生成建议区域[tx,ty,tl]
        scaledA_x = self.crop_resize(x, atten1 * 448)     # 在裁剪区域的位置
        # scaledA_x: (224, 224) size image

        conv5_4_A = self.b2.features[:-1](scaledA_x)  # 此时传入的图像是裁剪过后的图像
        pool5_A = self.feature_pool2(conv5_4_A)
        atten2 = self.apn2(conv5_4_A.view(-1, 512 * 14 * 14))
        scaledAA_x = self.crop_resize(scaledA_x, atten2 * 224) # 传入的位置是图像大小*注意力位置 

        pool5_AA = self.feature_pool2(self.b3.features[:-1](scaledAA_x))
        # 池化后的输出
        pool5 = pool5.view(-1, 512)
        pool5_A = pool5_A.view(-1, 512)
        pool5_AA = pool5_AA.view(-1, 512)

        """#Feature fusion
        scale123 = torch.cat([pool5, pool5_A, pool5_AA], 1)
        scale12 = torch.cat([pool5, pool5_A], 1)
        """

        logits1 = self.classifier1(pool5)
        logits2 = self.classifier2(pool5_A)
        logits3 = self.classifier3(pool5_AA)

        return [logits1, logits2, logits3], [conv5_4, conv5_4_A], [atten1, atten2], [scaledA_x, scaledAA_x]

# 注意力裁剪
class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        # logistic function k=10
        h = lambda x: 1. / (1. + torch.exp(-10. * x))
        # img_weight
        in_size = images.size()[2]
        # 生成 in_size*in_size 维度的 0~in_size 的矩阵 
        unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
        # 模拟 3 通道 x , y=x.t 的值
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            #    x, y = x.cuda(), y.cuda()
            x, y = x.to(device), y.to(device)

        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):  # 迭代通道
            '''
            loc=[[tx,ty,tl]
                [tx,ty,tl]
                [tx,ty,tl]]
            '''
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            # 限制图像的框
            tl = tl if tl > (in_size / 3) else in_size / 3
            tx = tx if tx > tl else tl      # 确保x的位置大于中间长度
            tx = tx if tx < in_size - tl else in_size - tl # 确保框在中间
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size - tl else in_size - tl
            # 宽度偏移
            w_off = int(tx - tl) if (tx - tl) > 0 else 0
            h_off = int(ty - tl) if (ty - tl) > 0 else 0
            # 宽度相对 x 坐标的位置
            w_end = int(tx + tl) if (tx + tl) < in_size else in_size
            h_end = int(ty + tl) if (ty + tl) < in_size else in_size
            # attention mask(判断点是否在APN的区域中)
            mk = (h(x - w_off) - h(x - w_end)) * (h(y - h_off) - h(y - h_end))
            
            # crop attended region
            xatt = images[i] * mk
            # attention region
            #print('xatt_shape:',xatt.shape)
            xatt_cropped = xatt[:, w_off:w_end, h_off:h_end] # 注意力裁剪区域
            #print('xatt_cropped',xatt_cropped.shape)
            before_upsample = Variable(xatt_cropped.unsqueeze(0)) # 增加一个维度
            #print('before_upsample shape',before_upsample.shape)
            xamp = F.interpolate(before_upsample, size=(224, 224), mode='bilinear', align_corners=True) # 双线性插值来放大 attend region
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)  # 保存图像和映射map用于backward==>使用 self.saved_variabkes 来获取
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        #print('grad_output',grad_output.shape)
        images, ret_tensor = self.saved_tensors[0], self.saved_tensors[1]# saved_variables 不用，使用saved.tensors
        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_() # 初始化为 0
        norm = -(grad_output * grad_output).sum(dim=1)     # 导数范数的负平方

        #         show_image(inputs.cpu().data[0])
        #         show_image(ret_tensor.cpu().data[0])
        #         plt.imshow(norm[0].cpu().numpy(), cmap='gray')
        
        # 初始化值
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size / 3 * 2)
        short_size = (in_size / 3)
        # x,y,l的导数形式
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size) + (x >= long_size) + (y < short_size) + (y >= long_size)) > 0).float() * 2 - 1
        #batch下的x,y,l导数形式
        mx_batch = torch.stack([mx.float()] * grad_output.size(0)) # 4*224*224
        # print(mx_batch.shape)
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            # mx_batch = mx_batch.cuda()
            # my_batch = my_batch.cuda()
            # ml_batch = ml_batch.cuda()
            # ret = ret.cuda()
            mx_batch = mx_batch.to(device)
            my_batch = my_batch.to(device)
            ml_batch = ml_batch.to(device)
            ret = ret.to(device)
        # 返回三个位置值
        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1) # (用范数*导数图来获取向哪边优化)长宽求和，获取位置
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret

# 自定义特征区域裁剪测层
class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """

    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)


if __name__ == '__main__':

    print(" [*] RACNN forward test...")
    x = torch.randn([2, 3, 448, 448])
    # 用image 来测试
    # path = '/data0/hwl_data/FGVC/CUB-1/train/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg'
    # x = Image.open(path).convert('RGB')
    # transform = transforms.Compose([
    #             transforms.Resize(448),
    #             transforms.RandomCrop([448, 448]),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #         ])
    # img_ten = transform(x).unsqueeze(0)
    #
    net = RACNN(num_classes=200)
    net = net.to(device)
    #[logits1, logits2, logits3], [conv5_4, conv5_4_A], [atten1, atten2], [scaledA_x, scaledAA_x]
    logits, conv5s, attens,scaleds = net(x.to(device))
    #logits, conv5s, attens,scaleds = net(img_ten.to(device))
    # modify_img = scaleds[0]
    # trans = transforms.ToPILImage()
    # img = trans(modify_img.squeeze(0))
    # img.save('1.jpg')
    # print(modify_img)
    print(" [*] logits[0]:", logits[0].size())
    from Loss import multitask_loss, pairwise_ranking_loss
    # 转为 GPU
    target_cls = torch.LongTensor([100, 150]).to(device)

    preds = []
    for i in range(len(target_cls)):
        pred = [logit[i][target_cls[i]] for logit in logits]
        preds.append(pred)
    loss_cls = multitask_loss(logits, target_cls)
    loss_rank = pairwise_ranking_loss(preds)
    print(" [*] Loss cls:", loss_cls)   # [tensor(5.3305, device='cuda:0', grad_fn=<NllLossBackward0>), tensor(5.2256, device='cuda:0', grad_fn=<NllLossBackward0>), tensor(5.3766, device='cuda:0', grad_fn=<NllLossBackward0>)]
    print(" [*] Loss rank:", loss_rank) # tensor(0.2027, device='cuda:0', grad_fn=<MeanBackward0>)

    print(" [*] Backward test")
    # loss = loss_cls + loss_rank
    # print('[*] loss:',loss)
    # loss.backward()
    # print(" [*] Backward done")
