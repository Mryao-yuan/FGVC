import argparse


CUDA_VISIBLE_DEVICES = "3"


# 参数设置
def set_arg():
    parser = argparse.ArgumentParser(description="NetModel config")
    # path
    parser.add_argument(
        "--data_path",
        default="/data0/hwl_data/FGVC/CUB/",
        type=str,
        help="path of dataset",
    )
    parser.add_argument(
        "--save_path",
        default="/data0/yaoyuan/FGVC/Net-Model",
        type=str,
        metavar="PATH",
        help="the path of file to save",
    )
    parser.add_argument(
        "--pth_path",
        default="/data0/yy_data/FGVC/MMAL-Net/Pretrained/resnet50-19c8e357.pth",
        type=str,
        help="path of pretrained  resnet model",
    )
    parser.add_argument(
        "--model_path",
        default="/data0/yy_data/FGVC/NetModel/checkpoint",
        type=str,
        help="Name of model",
    )
    # optim
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--lr_decay_rate", default=0.1, type=float)
    parser.add_argument("--lr_milestones", default=[60, 100], type=list)
    # dataset
    parser.add_argument("-set", default="CUB", type=str, help="dataset defalut cub")
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--eval_trainset", default=False, type=bool,help="Wether or not evaluate trainset")
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=6,
        type=int,
        metavar="N",
        help="mini-batch size (default: 6)",
    )
    # optim
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.01,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--save_interval",
        "-s",
        default=1,
        type=int,
        metavar="N",
        help=" (default: 10)",
    )
    parser.add_argument(
        "--evaluate-freq", default=10, type=int, help="the evaluation frequence"
    )
    parser.add_argument(
        "--resume",
        default="./checkpoint.pth.tar",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--init_lr", default=0.0001, type=float, help="learning rating")

    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    # dataset
    parser.add_argument(
        "--n_classes", default=200, type=int, help="the number of classes"
    )
    parser.add_argument(
        "--input_size", default=448, type=int, help="the number of samples per class"
    )
        
    return parser.parse_args()


if __name__ == "__main__":
    arg = set_arg()
