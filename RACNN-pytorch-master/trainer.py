from visual import Logger

import os
import os.path as osp
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
from data import CUB_loader
from data.CUB_loader import CUB200_loader, dataset_path, test_path

import cv2

from models import RACNN, pairwise_ranking_loss, multitask_loss
from models.RACNN import *
from utils import save_img
import wandb

# 初始化wandb的格式
# wandb.init(project='bird_200',name=time.strftime('%m-%d-%H:%M:%S'))
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument("--cuda", default=True, type=bool, help="use cuda to train")
parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
parser.add_argument("--batch_size", default=1, type=int, help="run for times")
# 参数通过 train.sh 指定为 cuda0
args = parser.parse_args()
decay_steps = [20, 40]  # based on epoch
# 输出类别 200类
net = RACNN(num_classes=200)
# 选择device
# if args.cuda and torch.cuda.is_available():
if torch.cuda.is_available():
    print(" [*] Set cuda: True")
    net = net.to(device)
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    print(" [*] Set cuda: False")
# 记录存储位置
# logger = Logger('./visual/' + 'RACNN_CUB200_9')
# 分类网络参数
# print(net.b1)
#  cls_params = list(net.module.b1.parameters()) + list(net.module.b2.parameters()) + list(
#     net.module.b3.parameters()) + list(net.module.classifier1.parameters()) + list(
#    net.module.classifier2.parameters()) + list(net.module.classifier3.parameters())

cls_params = (
    list(net.b1.parameters())
    + list(net.b2.parameters())
    + list(net.b3.parameters())
    + list(net.classifier1.parameters())
    + list(net.classifier2.parameters())
    + list(net.classifier3.parameters())
)

opt1 = optim.SGD(cls_params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
# 注意力建议网络参数
apn_params = list(net.apn1.parameters()) + list(net.apn2.parameters())
opt2 = optim.SGD(apn_params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
# 将tensor类型数据取小数点后n位
# def get_format_num(float_data, points=4):
#     num_format = "{num:.{points}f}"
#     num = num_format.format(num=float_data, points=points)
#     return float(num)


def train():
    log_train = {}
    net.train()

    conf_loss = 0
    loc_loss = 0

    print(" [*] Loading dataset...")
    batch_iterator = None

    # dataset_path = '/data0/hwl_data/FGVC/CUB/'
    # 数据载入
    trainset = CUB200_loader(dataset_path, split="train")
    trainloader = data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        collate_fn=trainset.CUB_collate,
        num_workers=4,
    )
    testset = CUB200_loader(dataset_path, split="test")
    testloader = data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        collate_fn=testset.CUB_collate,
        num_workers=4,
    )
    # 迭代
    test_sample, _ = next(iter(testloader))
    # 预训练特征注意力网络
    apn_iter, apn_epoch, apn_steps = pretrainAPN(trainset, trainloader)
    cls_iter, cls_epoch, cls_steps = 0, 0, 1
    switch_step = 0
    old_cls_loss, new_cls_loss = 2, 1
    old_apn_loss, new_apn_loss = 2, 1
    iteration = 0  # count the both of iteration
    epoch_size = len(trainset) // 4  # 每次迭代的大小
    cls_tol = 0
    apn_tol = 0
    batch_iterator = iter(trainloader)
    # 迭代500000 次或分类损失和apn损失误差足够小
    while (
        ((old_cls_loss - new_cls_loss) ** 2 > 1e-7)
        and ((old_apn_loss - new_apn_loss) ** 2 > 1e-7)
        and (iteration < 500000)
    ):
        # until the two type of losses no longer change
        print(" [*] Swtich optimize parameters to Class")
        while (cls_tol < 10) and (cls_iter % 5000 != 0):  # 分类损失收敛次数超过十次
            if (not batch_iterator) or (iteration % epoch_size == 0):
                batch_iterator = iter(trainloader)

            if cls_iter % epoch_size == 0:
                cls_epoch += 1
                if cls_epoch in decay_steps:
                    cls_steps += 1
                    adjust_learning_rate(opt1, 0.1, cls_steps, args.lr)

            old_cls_loss = new_cls_loss

            images, labels = next(batch_iterator)
            images, labels = Variable(images, requires_grad=True), Variable(labels)
            if args.cuda:
                # images, labels = images.cuda(), labels.cuda()
                images, labels = images.to(device), labels.to(device)

            t0 = time.time()
            # 获取分类结果
            logits, _, _, _ = net(images)

            opt1.zero_grad()
            new_cls_losses = multitask_loss(logits, labels)
            new_cls_loss = sum(new_cls_losses)  # 分类损失
            # 分类损失反向传播
            new_cls_loss.backward()
            opt1.step()
            t1 = time.time()
            # 损失误差很小，接近收敛
            if (old_cls_loss - new_cls_loss) ** 2 < 1e-6:
                cls_tol += 1
            else:
                cls_tol = 0

            # log_train['iter'] = iteration+1
            # log_train['cls_loss'] =  new_cls_loss.item()
            # log_train['cls_loss1'] =  new_cls_loss[0].item()
            # log_train['cls_loss12'] =  new_cls_loss[1].item()
            # log_train['cls_loss123'] =  new_cls_loss[2].item()

            iteration += 1
            cls_iter += 1
            if (cls_iter % 20) == 0:
                print(
                    " [*] cls_epoch[%d], Iter %d || cls_iter %d || cls_loss: %.4f || Timer: %.4fsec"
                    % (cls_epoch, iteration, cls_iter, new_cls_loss.item(), (t1 - t0))
                )
        # 迭代图像
        images, labels = next(batch_iterator)
        # if args.cuda:
        if torch.cuda.is_available():
            images, labels = images.to(device), labels.to(device)
            print("torch.cuda  is available")
        logits, _, _, _ = net(images)
        preds = []
        for i in range(len(labels)):
            pred = [logit[i][labels[i]] for logit in logits]
            preds.append(pred)
        new_apn_loss = pairwise_ranking_loss(preds)
        # logger.scalar_summary('rank_loss', new_apn_loss.item(), iteration + 1)
        log_train["rank_loss"] = new_apn_loss.item()
        iteration += 1
        # cls_iter += 1
        test(testloader, iteration)  # 测试
        # continue
        print(" [*] Swtich optimize parameters to APN")
        switch_step += 1

        while (apn_tol < 10) and apn_iter % 5000 != 0:
            if (not batch_iterator) or (iteration % epoch_size == 0):
                batch_iterator = iter(trainloader)

            if apn_iter % epoch_size == 0:
                apn_epoch += 1
                if apn_epoch in decay_steps:
                    apn_steps += 1
                    adjust_learning_rate(opt2, 0.1, apn_steps, args.lr)

            old_apn_loss = new_apn_loss

            images, labels = next(batch_iterator)
            images, labels = Variable(images, requires_grad=True), Variable(labels)
            if torch.cuda.is_available():
                # images, labels = images.cuda(), labels.cuda()
                images, labels = images.to(device), labels.to(device)

            t0 = time.time()
            logits, _, _, _ = net(images)

            opt2.zero_grad()
            preds = []
            for i in range(len(labels)):
                pred = [logit[i][labels[i]] for logit in logits]
                preds.append(pred)
            new_apn_loss = pairwise_ranking_loss(preds)
            new_apn_loss.backward()
            opt2.step()
            t1 = time.time()

            if (old_apn_loss - new_apn_loss) ** 2 < 1e-6:
                apn_tol += 1
            else:
                apn_tol = 0

            # logger.scalar_summary('rank_loss', new_apn_loss.item(), iteration + 1)
            iteration += 1
            apn_iter += 1
            if (apn_iter % 20) == 0:
                print(
                    " [*] apn_epoch[%d], Iter %d || apn_iter %d || apn_loss: %.4f || Timer: %.4fsec"
                    % (apn_epoch, iteration, apn_iter, new_apn_loss.item(), (t1 - t0))
                )

        switch_step += 1

        images, labels = next(batch_iterator)
        if torch.cuda.is_available():
            # images, labels = images.cuda(), labels.cuda()
            images, labels = images.to(device), labels.to(device)
        logits, _, _, _ = net(images)
        new_cls_losses = multitask_loss(logits, labels)
        new_cls_loss = sum(new_cls_losses)
        # logger.scalar_summary('cls_loss', new_cls_loss.item(), iteration + 1)
        log_train["cla_loss"] = new_cls_loss.item()
        # wandb.log(log_train)
        iteration += 1
        cls_iter += 1
        apn_iter += 1
        test(testloader, iteration)
        # _, _, _, crops = net(test_sample)
        _, _, _, crops = net(test_sample.to(device))
        x1, x2 = crops[0].data, crops[1].data
        # visualize cropped inputs
        # save_img(x1, path=f'samples/iter_{iteration}@2x.jpg', annotation=f'loss = {avg_loss:.7f}, step = {iteration}')
        # save_img(x2, path=f'samples/iter_{iteration}@4x.jpg', annotation=f'loss = {avg_loss:.7f}, step = {iteration}')
        # torch.save(net.state_dict, 'ckpt/RACNN_vgg_CUB200_iter%d.pth' % iteration)


# 特征注意网络的预训练
def pretrainAPN(trainset, trainloader):
    log_apn = {}
    epoch_size = len(trainset) // 4  # batch_size = 4
    apn_steps, apn_epoch = 1, -1

    batch_iterator = iter(trainloader)
    for _iter in range(0, 2000):
        iteration = _iter
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(trainloader)

        if iteration % epoch_size == 0:
            apn_epoch += 1
            if apn_epoch in decay_steps:
                apn_steps += 1
                adjust_learning_rate(opt2, 0.1, apn_steps, args.lr)
        # 获取图像和标签
        images, labels = next(batch_iterator)
        # 对图像求导
        images, labels = Variable(images, requires_grad=True), Variable(labels)
        # GPU 加速
        # if args.cuda:
        if torch.cuda.is_available():
            # images, labels = images.cuda(), labels.cuda()
            images, labels = images.to(device), labels.to(device)

        t0 = time.time()
        # 获取卷积核注意力特征
        _, conv5s, attens, _ = net(images)  # 优化区域

        opt2.zero_grad()
        # search regions with the highest response value in conv5
        weak_loc = []
        for i in range(len(conv5s)):  # 两个不同尺度的迭代最高响应值
            # 设定tl为边长的1/3(tx,ty,tl)
            loc_label = torch.ones([images.size(0), 3]) * 0.33  # tl = 0.25, fixed
            resize = 448
            if i >= 1:  # 裁剪后一次后的区域
                resize = 224
            # if args.cuda:
            if torch.cuda.is_available():
                # loc_label = loc_label.cuda()
                loc_label = loc_label.to(device)

            for j in range(
                images.size(0)
            ):  # 迭代batch（4张图片一次）  torch.Size([4, 3, 448, 448])
                response_map = conv5s[i][j]  # torch.Size([512, 28, 28])
                response_map = F.interpolate(
                    response_map.unsqueeze(0), size=[resize, resize]
                )  # 上采样
                # print('response_map:',response_map.shape)
                response_map = response_map.mean(0)  # 对通道求平均
                # print('response_map_mean:',response_map.shape)
                rawmaxidx = response_map.view(-1).max(0)[1]  # 特征图张开获取行最大的下标
                idx = []
                for d in list(response_map.size())[::-1]:
                    idx.append(rawmaxidx % d)  #  在原始图上的横坐标
                    rawmaxidx = rawmaxidx / d  #  下一个行
                loc_label[j, 0] = (idx[1].float() + 0.5) / response_map.size(
                    0
                )  # 将中心点坐标放入local_label
                loc_label[j, 1] = (idx[0].float() + 0.5) / response_map.size(1)
            weak_loc.append(loc_label)
        weak_loss1 = F.smooth_l1_loss(
            attens[0], weak_loc[0]
        )  # 计算边框的损失(卷积获取的区域与本地基于相应区域进行损失计算)
        weak_loss2 = F.smooth_l1_loss(attens[1], weak_loc[1])
        apn_loss = weak_loss1 + weak_loss2
        apn_loss.backward()  # 更新裁剪区域
        opt2.step()  # 梯度优化
        t1 = time.time()

        if (iteration % 20) == 0:
            print(
                " [*] pre_apn_epoch[%d], || pre_apn_iter %d || pre_apn_loss: %.4f || Timer: %.4fsec"
                % (apn_epoch, iteration, apn_loss.item(), (t1 - t0))
            )
        log_apn["apn-epoch"] = apn_epoch
        log_apn["apn-iter"] = iteration

        # log_apn['apn-loss'] = get_format_num(apn_loss.item())
        log_apn["apn-loss"] = apn_loss.item()
        # wandb.log(log_apn)

        # logger.scalar_summary('pre_apn_loss', apn_loss.item(), iteration + 1)
    # apn_iter apn_epoch apn_steps
    return 2000, apn_epoch, apn_steps


# 验证准确率
def test(testloader, iteration):
    log_val = {}
    net.eval()
    with torch.no_grad():
        corrects1 = 0
        corrects2 = 0
        corrects3 = 0
        cnt = 0
        test_cls_losses = []
        test_apn_losses = []
        for test_images, test_labels in testloader:
            if torch.cuda.is_available():
                # test_images = test_images.cuda()
                # test_labels = test_labels.cuda()
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
            cnt += test_labels.size(0)

            logits, _, _, _ = net(test_images)

            preds = []
            for i in range(len(test_labels)):
                pred = [logit[i][test_labels[i]] for logit in logits]
                preds.append(pred)
            # 测试分类损失
            test_cls_losses = multitask_loss(logits, test_labels)
            # 测试注意力损失
            test_apn_loss = pairwise_ranking_loss(preds)

            test_cls_losses.append(sum(test_cls_losses))
            test_apn_losses.append(test_apn_loss)
            _, predicted1 = torch.max(logits[0], 1)
            correct1 = (predicted1 == test_labels).sum()
            corrects1 += correct1
            _, predicted2 = torch.max(logits[1], 1)
            correct2 = (predicted2 == test_labels).sum()
            corrects2 += correct2
            _, predicted3 = torch.max(logits[2], 1)
            correct3 = (predicted3 == test_labels).sum()
            corrects3 += correct3

        test_cls_losses = torch.stack(test_cls_losses).mean()
        test_apn_losses = torch.stack(test_apn_losses).mean()
        accuracy1 = corrects1.float() / cnt
        accuracy2 = corrects2.float() / cnt
        accuracy3 = corrects3.float() / cnt
        # logger.scalar_summary('test_cls_loss', test_cls_losses.item(), iteration + 1)
        # logger.scalar_summary('test_rank_loss', test_apn_losses.item(), iteration + 1)
        # logger.scalar_summary('test_acc1', accuracy1.item(), iteration + 1)
        # logger.scalar_summary('test_acc2', accuracy2.item(), iteration + 1)
        # logger.scalar_summary('test_acc3', accuracy3.item(), iteration + 1)
        print(
            " [*] Iter %d || Test accuracy1: %.4f, Test accuracy2: %.4f, Test accuracy3: %.4f"
            % (iteration, accuracy1.item(), accuracy2.item(), accuracy3.item())
        )
        log_val["iter"] = iteration + 1
        log_val["test_cls_loss"] = test_cls_losses.item()
        log_val["test_rank_loss"] = test_apn_losses.item()
        log_val["test_acc1"] = accuracy1.item()
        log_val["test_acc2"] = accuracy1.item()
        log_val["test_acc3"] = accuracy1.item()
        # wandb.log(log_val)

    net.train()


# 调整学习率
def adjust_learning_rate(optimizer, gamma, steps, _lr):
    lr = _lr * (gamma ** (steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    train()
    # print(get_format_num(1.2324234234234, points=5))
    print(" [*] Train done")
