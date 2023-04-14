import os


def write_log(root="/data0/yy_data/FGVC/MMAL-Net/checkpoint", file="train.txt", obj={}):
    with open(os.path.join(root, file), "w+") as wf:
        for key, value in obj.items():
            wf.writelines(key + ": " + str(value) + "\n")
