import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import time
from utils.config import set_arg, CUDA_VISIBLE_DEVICES


from Dataset import dataset
import os
from Model.MainNet import MainNet
from utils.config import set_arg
from utils.write_log import write_log
from utils.auto_load_resume import auto_load_resume
from utils.train_mode import train
from utils.write_config import write_config


def main():
    # 参数配置
    args = set_arg()
    # 配置参数写入文件
    write_config(args, "config.txt")
    # 加载数据
    trainloader, testloader = dataset.read_dataset(
        args.input_size, args.batch_size, args.data_path, args.set
    )

    # 定义模型
    model = MainNet(num_class=args.n_classes)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    # 参数
    parameters = model.parameters()

    # 加载checkpoint
    save_path = os.path.join(args.model_path, args.set)
    if os.path.exists(save_path):
        start_epoch, lr = auto_load_resume(model, save_path, status="train")
        assert start_epoch < args.epochs
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = args.init_lr

    # define optimizers
    optimizer = torch.optim.SGD(
        parameters, lr=lr, momentum=0.9, weight_decay=args.weight_decay
    )

    model = model.cuda()  # 部署在GPU

    scheduler = MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_decay_rate
    )

    # 保存config参数信息
    time_str = time.strftime("%Y%m%d-%H%M%S")

    # 文件复制
    # shutil.copy(
    #     "./config.py", os.path.join(save_path, "config", "{}config.py".format(time_str))
    # )

    # 开始训练
    train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=save_path,
        start_epoch=start_epoch,
        end_epoch=args.epochs,
        save_interval=args.save_interval,
        set=args.set,
        eval_trainset=args.eval_trainset,
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    main()
    ###############
