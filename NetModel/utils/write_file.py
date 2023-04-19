import os


def write_config(args, file):
    # 当前文件夹的上上层目录中
    path_ = os.path.dirname(os.path.dirname(__file__))  # __file__ 当前文件
    args_Dict = args.__dict__
    path = os.path.join(path_, file)
    with open(path, "w") as f:
        for key, value in args_Dict.items():
            f.writelines(key + " : " + str(value) + "\n")

    print("config parameter has beem written in {} success!".format(path))


def write_log(file, epoch, obj):
    path = os.path.dirname(os.path.dirname(__file__))
    root = os.path.join(path, file)
    with open(root, "w") as wf:
        wf.writelines("[ {} ] \n".format(epoch))
        for key, value in obj.items():
            wf.writelines("\t" + key + ": " + str(value) + "\n  ")
    print("eval data has been written in {}".format(root))
