from utils.write_file import write_log, write_config
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.eval_mode import eval


def train(
    model,
    trainloader,
    testloader,
    criterion,
    optimizer,
    scheduler,
    save_path,
    start_epoch,
    end_epoch,
    save_interval,
    set,
    eval_trainset,
):
    r"""
    定义训练模型方式
    """
    max_eval_acc = 0.0
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        print("Training %d epoch" % epoch)
        # 学习率的调整
        lr = next(iter(optimizer.param_groups))["lr"]
        for i, data in enumerate(tqdm(trainloader)):
            if set == "CUB":
                # 本网络未使用鸟类数据集的边界框和scale
                images, labels, _, _ = data
            # images, labels = data
            else:
                # 查看不同数据集的返回类型是否一致
                images, labels = data
            # images[batch channel height width]
            images, labels = images.cuda(), labels.cuda()
            # 梯度清零
            optimizer.zero_grad()
            # object :[batch h w channel]
            (object_img, bboxes_list, output) = model(images, epoch, i)
            # 仅仅使用VGG
            # output = model(images, epoch, i)
            # 计算损失
            cls_loss = criterion(output, labels)

            cls_loss.backward()

            optimizer.step()

        scheduler.step()
        # evaluation every epoch
        if eval_trainset:
            eval_traindatase_data = {}
            loss_avg, local_accuracy = eval(
                model, trainloader, criterion, "train", save_path, epoch
            )
            print("[Train set] cls accuary:{:.2f}%".format(100.0 * local_accuracy))
            eval_traindatase_data["local_acc"] = round(local_accuracy * 100, 2)
            eval_traindatase_data["loss_avg"] = round(loss_avg, 2)
            write_log(file="train.txt", epoch=epoch, obj=eval_traindatase_data)
            # 记录最大的准确率值及其epoch
            # if raw_accuracy > train_max[0][0]:
            #     train_max[0][0] = round(raw_accuracy, 2)
            #     train_max[1][0] = epoch
            # if local_accuracy > train_max[0][1]:
            #     train_max[0][0] = round(local_accuracy, 2)
            #     train_max[1][0] = epoch
            # write accuracy to train.txt
            # train_obj[str(epoch) + ") raw_accuracy\t"] = (
            #     str(round(100 * raw_accuracy, 2)) + "%"
            # )
            # train_obj[str(epoch) + ") local_accuracy"] = (
            #     str(round(100 * local_accuracy, 2)) + "%"
            # )

            # tensorboard
            with SummaryWriter(
                log_dir=os.path.join(save_path, "log"), comment="train"
            ) as writer:
                writer.add_scalar("Train/learning rate", lr, epoch)
                writer.add_scalar("Train/local_accuracy", local_accuracy, epoch)
                writer.add_scalar("Train/loss_avg", loss_avg, epoch)

        # eval testset
        eval_testdatase_data = {}
        loss_avg, local_accuracy = eval(
            model, testloader, criterion, "test", save_path, epoch
        )
        print("[Test set] cls accuary:{:.2f}%".format(100.0 * local_accuracy))
        eval_testdatase_data["local_acc"] = round(local_accuracy * 100, 2)
        eval_testdatase_data["loss_avg"] = round(loss_avg, 2)
        write_log(file="test.txt", epoch=epoch, obj=eval_testdatase_data)
        # 记录当前最大的准确率
        if local_accuracy > max_eval_acc:
            max_eval_acc = local_accuracy
            save_status = True
        else:
            save_status = False
        # 记录最大的准确率值及其epoch
        # if raw_accuracy > test_max[0][0]:
        #     test_max[0][0] = round(raw_accuracy, 2)
        #     test_max[1][0] = epoch
        # if local_accuracy > test_max[0][1]:
        #     test_max[0][0] = round(local_accuracy, 2)
        #     test_max[1][0] = epoch
        # write accuracy to test.txt
        # test_obj[str(epoch) + ") raw_accuracy"] = str(round(100 * raw_accuracy)) + "%"
        # test_obj[str(epoch) + ") local_accuracy"] = (
        #     str(round(100 * local_accuracy)) + "%"
        # )
        # write_log(file="test.txt", obj=test_obj)

        # tensorboard
        with SummaryWriter(
            log_dir=os.path.join(save_path, "log"), comment="test"
        ) as writer:
            writer.add_scalar("Test/local_accuracy", local_accuracy, epoch)
            writer.add_scalar("Test/loss_avg", loss_avg, epoch)

        # save checkpoint
        if (epoch % save_interval == 0) and (save_status) or (epoch == end_epoch):
            print("Saving checkpoint")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "learning_rate": lr,
                },
                os.path.join(save_path, "epoch" + str(epoch) + ".pth"),
            )

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [
            os.path.basename(path)
            for path in glob.glob(os.path.join(save_path, "*.pth"))
        ]

        # 判断是否完成指定epoch
        if len(checkpoint_list) == 200:
            # 获取checkpoint的对应epoch
            idx_list = [
                int(name.replace("epoch", "").replace(".pth", ""))
                for name in checkpoint_list
            ]
            # for i in idx_list:
            #     if i != test_max[1, 0] | i != test_max[2, 0]:
            #         os.remove(os.path.join(save_path, "epoch" + str(i) + ".pth"))
            # min_idx = min(idx_list)
        # 移除最小epoch的save文件
        # os.remove(os.path.join(save_path, "epoch" + str(min_idx) + ".pth"))
