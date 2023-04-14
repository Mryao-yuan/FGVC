import os


def write_config(args, file):
    args_Dict = args.__dict__
    # par_pth = os.path.dirname(os.getcwd())
    path = os.path.join(os.getcwd(), file)
    with open(path, "a+") as f:
        for key, value in args_Dict.items():
            f.writelines(key + " : " + str(value) + "\n")

    print("config parameter has beem written in {} success!".format(path))
