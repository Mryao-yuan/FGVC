import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
import numpy as np
from torch import nn
from skimage import measure
import cv2
import os.path as osp
import os

import torch.nn.functional as F


# class DDT(object):
#     def __init__(self, use_cuda=False):
#         # 加载预训练模型
#         if not torch.cuda.is_available():
#             self.use_cuda = False
#         else:
#             self.use_cuda = use_cuda
#         print("use_cuda = %s" % str(self.use_cuda))
#         # 载入预训练模型
#         if self.use_cuda:
#             self.pretrained_feature_model = (
#                 models.vgg19(pretrained=True).features
#             ).cuda()
#         else:
#             self.pretrained_feature_model = models.vgg19(pretrained=True).features
#         # 标准化
#         self.normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )
#         # 类型更改
#         self.totensor = transforms.ToTensor()

#     def fit(self, traindir):
#         train_dataset = ImageSet(traindir, resize=1000)

#         descriptors = np.zeros((1, 512))

#         for index in range(len(train_dataset)):
#             print("processing " + str(index) + "th training images.")
#             image = train_dataset[index]
#             h, w = image.shape[:2]
#             image = self.normalize(self.totensor(image)).view(1, 3, h, w)
#             if self.use_cuda:
#                 image = image.cuda()
#             output = self.pretrained_feature_model(image)[0, :]
#             output = output.view(512, output.shape[1] * output.shape[2])
#             output = output.transpose(0, 1)
#             descriptors = np.vstack((descriptors, output.detach().cpu().numpy().copy()))
#             del output

#         descriptors = descriptors[1:]

#         # 计算descriptor均值，并将其降为0
#         descriptors_mean = sum(descriptors) / len(descriptors)
#         descriptors_mean_tensor = torch.FloatTensor(descriptors_mean)
#         pca = PCA(n_components=1)  # 保留 1 个特征
#         pca.fit(descriptors)
#         trans_vec = pca.components_[0]
#         return trans_vec, descriptors_mean_tensor

#     def co_locate(self, testdir, savedir, trans_vector, descriptor_mean_tensor):
#         r"""
#         使用图像集合定位
#         """
#         test_dataset = ImageSet(testdir, resize=1000)
#         if self.use_cuda:
#             descriptor_mean_tensor = descriptor_mean_tensor.cuda()
#         for index in range(len(test_dataset)):
#             image = test_dataset[index]
#             origin_image = image.copy()
#             origin_height, origin_width = origin_image.shape[:2]
#             image = self.normalize(self.totensor(image)).view(
#                 1, 3, origin_height, origin_width
#             )
#             if self.use_cuda:
#                 image = image.cuda()
#             featmap = self.pretrained_feature_model(image)[0, :]
#             h, w = featmap.shape[1], featmap.shape[2]
#             featmap = featmap.view(512, -1).transpose(0, 1)
#             featmap -= descriptor_mean_tensor.repeat(featmap.shape[0], 1)
#             features = featmap.detach().cpu().numpy()
#             del featmap

#             P = np.dot(trans_vector, features.transpose()).reshape(h, w)

#             mask = self.max_conn_mask(P, origin_height, origin_width)
#             bboxes = self.get_bboxes(mask)

#             mask_3 = np.concatenate(
#                 (
#                     np.zeros((2, origin_height, origin_width), dtype=np.uint16),
#                     mask * 255,
#                 ),
#                 axis=0,
#             )

#             """ 将原图同mask相加并展示
#             mask_3 = np.transpose(mask_3, (1, 2, 0))
#             mask_3 = origin_image + mask_3
#             mask_3[mask_3[:, :, 2] > 254, 2] = 255
#             mask_3 = np.array(mask_3, dtype=np.uint8)
#             """

#             """draw bboxes
#             for x, y, w, h in bboxes:
#                 cv2.rectangle(mask_3, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             """
#             print("save the " + str(index) + "th image. ")
#             cv2.imwrite(osp.join(savedir, str(index) + ".jpg"), mask_3)

#     def max_conn_mask(self, P, origin_height, origin_width):
#         h, w = P.shape[0], P.shape[1]
#         highlight = np.zeros(P.shape)
#         # 掩码操作

#         highlight[P > 0] = 1

#         # 寻找最大的全联通分量
#         # labels = measure.label(highlight, neighbors=4, background=0)
#         labels = measure.label(highlight, connectivity=1, background=0)
#         props = measure.regionprops(labels)
#         max_index = 0
#         for i in range(len(props)):
#             if props[i].area > props[max_index].area:
#                 # 获取最大区域的下标
#                 max_index = i
#         max_prop = props[max_index]
#         highlights_conn = np.zeros(highlight.shape)
#         for each in max_prop.coords:
#             highlights_conn[each[0]][each[1]] = 1

#         # 最近邻插值：
#         highlight_big = cv2.resize(
#             highlights_conn,
#             (origin_width, origin_height),
#             interpolation=cv2.INTER_NEAREST,
#         )

#         highlight_big = np.array(highlight_big, dtype=np.uint16).reshape(
#             1, origin_height, origin_width
#         )
#         # highlight_3 = np.concatenate((np.zeros((2, origin_height, origin_width), dtype=np.uint16), highlight_big * 255), axis=0)
#         return highlight_big

#     def get_bboxes(self, bin_img):
#         img = np.squeeze(bin_img.copy().astype(np.uint8), axis=(0,))

#         # _, contours, hierarchy = cv2.findContours(
#         #     img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         # )
#         contours, hierarchy = cv2.findContours(
#             img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )

#         bboxes = []
#         for c in contours:
#             # find bounding box coordinates
#             # 现计算出一个简单的边界框
#             # c = np.squeeze(c, axis=(1,))
#             rect = cv2.boundingRect(c)
#             bboxes.append(rect)

#         return bboxes


def get_bboxes(bin_img):
    img = np.squeeze(bin_img.copy().astype(np.uint8), axis=(0,))
    # 获取轮廓
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for c in contours:
        # find bounding box coordinates
        # 现计算出一个简单的边界框
        # c = np.squeeze(c, axis=(1,))
        rect = cv2.boundingRect(c)
        bboxes.append(rect)
    return bboxes


def fit(image, pretrain_module):
    descriptors = np.zeros((1, 512))
    for index in range(image.shape[0]):
        # print("processing " + str(index) + "th training images.")
        img = image[index]
        # h, w = img.shape[:2] w,h,c 模式
        h, w = img.shape[1:]  # c,w,h 模式
        img = img.view(1, 3, h, w)
        output = pretrain_module(img)[0, :]  # get feature of image
        output = output.view(512, output.shape[1] * output.shape[2])
        output = output.transpose(0, 1)
        descriptors = np.vstack((descriptors, output.detach().cpu().numpy().copy()))
        del output

    descriptors = descriptors[1:]
    # 计算descriptor均值，并将其降为0
    descriptors_mean = sum(descriptors) / len(descriptors)
    descriptors_mean_tensor = torch.FloatTensor(descriptors_mean)
    pca = PCA(n_components=1)  # 保留 1 个特征
    if np.nan in descriptors:
        print(descriptors)
    pca.fit(descriptors)
    trans_vec = pca.components_[0]
    return trans_vec, descriptors_mean_tensor


def max_conn_mask(P, origin_height, origin_width):
    """通过描述符来获取连通区域的对象坐标"""
    h, w = P.shape[0], P.shape[1]
    highlight = np.zeros(P.shape)
    # 掩码操作
    highlight[P > 0] = 1
    # 寻找最大的全联通分量
    labels = measure.label(highlight, connectivity=1, background=0)
    props = measure.regionprops(labels)
    max_index = 0
    for i in range(len(props)):
        if props[i].area > props[max_index].area:
            # 获取最大区域的下标
            max_index = i
    # 存在为 0 的情况(不减少) 相关性都为 0
    if max_index == 0:
        highlights_conn = np.ones(highlight.shape)
    else:
        max_prop = props[max_index]
        highlights_conn = np.zeros(highlight.shape)
        for each in max_prop.coords:
            highlights_conn[each[0]][each[1]] = 1
    # 最近邻插值：
    highlight_big = cv2.resize(
        highlights_conn,
        (origin_width, origin_height),
        interpolation=cv2.INTER_NEAREST,
    )
    highlight_big = np.array(highlight_big, dtype=np.uint16).reshape(
        1, origin_height, origin_width
    )
    return highlight_big


def co_locate(images, pretrained_model, trans_vector, descriptor_mean_tensor):
    r"""
    使用图像集合定位
    """
    # descriptor_mean_tensor = descriptor_mean_tensor.cuda()

    crop_img_list = np.zeros_like(images.cpu().numpy())
    bboxes_list = []
    for index in range(images.shape[0]):
        image = images[index]
        origin_image = image.clone()

        # origin_height, origin_width = origin_image.shape[:2]
        origin_height, origin_width = origin_image.shape[1:]
        image = image.view(1, 3, origin_height, origin_width)
        # image = image.cuda()
        image = image
        featmap = pretrained_model(image)[0, :]
        # 14 *14
        h, w = featmap.shape[1], featmap.shape[2]
        featmap = featmap.view(512, -1).transpose(0, 1)
        featmap -= descriptor_mean_tensor.repeat(featmap.shape[0], 1).cuda()
        features = featmap.detach().cpu().numpy()
        del featmap
        # 坐标关系矩阵
        P = np.dot(trans_vector, features.transpose()).reshape(h, w)
        mask = max_conn_mask(P, origin_height, origin_width)
        bboxes = get_bboxes(mask)
        bboxes_list.append(bboxes)
        # mask_3 = np.concatenate(
        #     (
        #         np.zeros((2, origin_height, origin_width), dtype=np.uint16),
        #         mask * 255,
        #     ),
        #     axis=0,
        # )
        for x, y, w, h in bboxes:
            #  h,w,c -> c,h,w 并更改 tensor
            # img_tensor = origin_image.permute(2, 0, 1).float()
            img_tensor = origin_image.float()
            # print(img_tensor.shape) torch.Size([3, 448, 448])
            crop_img = img_tensor[:, y : (y + h), x : (x + w)]
            # print("crop img ", crop_img.shape) torch.Size([3, 448, 448])
            new_img = F.interpolate(
                crop_img.unsqueeze(0),
                (origin_height, origin_height),
                mode="bilinear",
                align_corners=False,
            )
        # new_img = np.array(new_img.squeeze(0).permute(1, 2, 0))
        new_img = np.array(new_img.squeeze(0).cpu())
        crop_img_list[index, :, :, :] = np.array(new_img, dtype=np.uint8)
    # [batch h w channel]
    return crop_img_list, bboxes_list

    """
    # 将原图同mask相加并展示
    mask_3 = np.transpose(mask_3, (1, 2, 0))
    mask_3 = origin_image + mask_3
    mask_3[mask_3[:, :, 2] > 254, 2] = 255
    mask_3 = np.array(mask_3, dtype=np.uint8)
    # draw bboxes
    for x, y, w, h in bboxes:
         cv2.rectangle(mask_3, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print("save the " + str(index) + "th image. ")
    cv2.imwrite(osp.join(savedir, str(index) + ".jpg"), mask_3) 
    """


class DDT_module(nn.Module):
    r"""主要功能找到输入一个batch图像的object,返回原始图像插值后的数据"""

    def __init__(self, model):
        super(DDT_module, self).__init__()
        self.pretain_module = model

    def forward(self, x):
        trans_vectors, descriptor_means = fit(x, self.pretain_module)
        x, bboxes_list = co_locate(
            x, self.pretain_module, trans_vectors, descriptor_means
        )
        return x, bboxes_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traindir",
        type=str,
        default="./data/bird",
        help="the train data folder path.",
    )
    parser.add_argument(
        "--testdir", type=str, default="./data/bird", help="the test data folder path."
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="./data/result/bird/resnet50_layer3_1",
        help="the final result saving path. ",
    )
    parser.add_argument("--gpu", type=str, default="True", help="cuda device to run")
    parser.add_argument(
        "--pth",
        type=str,
        default="/data0/yy_data/FGVC/MMAL-Net/Pretrained/resnet50-19c8e357.pth",
        help="pretrain_path of resnet",
    )
    args = parser.parse_args()

    if not osp.exists(args.savedir):
        os.makedirs(args.savedir)
    if args.gpu != "":
        use_cuda = True
    else:
        use_cuda = False
    # ddt = DDT_self(use_cuda=use_cuda, pth=args.pth)
    # trans_vectors, descriptor_means = ddt.fit(args.traindir)
    # ddt.co_locate(args.testdir, args.savedir, trans_vectors, descriptor_means)
