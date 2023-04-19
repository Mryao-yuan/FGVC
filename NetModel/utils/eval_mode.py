import torch
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import numpy as np


def eval(model, testloader, criterion, status, save_path, epoch):
    model.eval()
    print("Evaluating")
    # raw_loss_sum = 0
    # windowscls_loss_sum = 0
    # total_loss_sum = 0
    # raw_correct = 0

    local_loss_sum = 0
    local_correct = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            images, labels, _, _ = data
            # images,labels = data
            images = images.cuda()
            labels = labels.cuda()
            # object :[batch h w channel]
            object_imgs, bboxes_list, output = model(images, epoch, i, status)
            # output = model(images, epoch, i, status)
            # 单个 epoch 的损失
            local_loss = criterion(output, labels)
            # 损失累加
            local_loss_sum += local_loss.item()
            # local
            pred = output.max(1, keepdim=True)[1]  # 获取最大值的下标
            # 计算判断正确的数量
            local_correct += pred.eq(labels.view_as(pred)).sum().item()

            # raw branch tensorboard
            # if i == 0:
            #     if set == "CUB":
            #         box_coor = resized_coor[:vis_num].numpy()
            #         pred_coor = coordinates[:vis_num].cpu().numpy()
            #         with SummaryWriter(
            #             log_dir=os.path.join(save_path, "log"), comment=status + "raw"
            #         ) as writer:
            #             cat_imgs = []
            #             for j, coor in enumerate(box_coor):
            #                 img = image_with_boxes(images[j], [coor])
            #                 img = image_with_boxes(
            #                     img, [pred_coor[j]], color=(0, 255, 0)
            #                 )
            #                 cat_imgs.append(img)
            #             cat_imgs = np.concatenate(cat_imgs, axis=1)
            #             writer.add_images(
            #                 status + "/" + "raw image with boxes",
            #                 cat_imgs,
            #                 epoch,
            #                 dataformats="HWC",
            #             )

            # object branch tensorboard
            # 暂时先不进行可视化
            # 问题：可视化的数据是经过归一化后的操作,影响正常观看
            # if i == 0:
            #     with SummaryWriter(
            #         log_dir=os.path.join(save_path, "log"), comment=status + "object"
            #     ) as writer:
            #         writer.add_images(
            #             status + "/" + "object image",
            #             object_imgs,
            #             epoch,
            #             # dataformats="NCHW",
            #         )
    loss_avg = local_loss_sum / (i + 1)
    local_accuracy = local_correct / len(testloader.dataset)

    return loss_avg, local_accuracy
