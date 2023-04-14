import os
import torch
from utils.config import set_arg
from collections import OrderedDict


def auto_load_resume(model, path, status):
    r"""
    加载保存的pth文件,并且加载效果最好的pth
    """
    args = set_arg()
    if status == "train":
        pth_files = os.listdir(path)
        # 之前的迭代次数
        nums_epoch = [
            int(name.replace("epoch", "").replace(".pth", ""))
            for name in pth_files
            if ".pth" in name
        ]
        if len(nums_epoch) == 0:
            return 0, args.init_lr
        else:
            max_epoch = max(nums_epoch)
            pth_path = os.path.join(path, "epoch" + str(max_epoch) + ".pth")
            print("Load model from", pth_path)
            # 载入权重文件
            checkpoint = torch.load(pth_path)
            # 字典形式存储权重文件
            new_state_dict = OrderedDict()
            """将checkpoint文件写入文本中
            file = open(os.path.join(path, "checkpoint.txt"), "w")
            for index, value in checkpoint.items():
                file.writelines(index + " : " + str(value) + "\n")
                print(index) : epoch,model_state_dict,learning_rate
            print("checkpoint:", checkpoint)
            """

            # k:pretrained_model.conv1.weight,pretrained_model.bn1.weight
            # v:value of the weight
            for k, v in checkpoint["model_state_dict"].items():
                new_state_dict[k] = v
                # 表明从第7个key值字符取到最后一个字符，正好去掉了module.
                name = k[7:]  # remove `module.` 载入先前的模型参数，故需要将先前的模型特有信息去除
                new_state_dict[name] = v
            # 出现载入键值不对的情况==>strict=False
            model.load_state_dict(new_state_dict, strict=False)
            epoch = checkpoint["epoch"]
            lr = checkpoint["learning_rate"]
            print("Resume from %s" % pth_path)
            return epoch, lr
    elif status == "test":
        print("Load model from", path)
        checkpoint = torch.load(path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            if "module." == k[:7]:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        epoch = checkpoint["epoch"]
        print("Resume from %s" % path)
        return epoch
