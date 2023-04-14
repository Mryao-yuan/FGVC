import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import imageio.v2 as imageio


class CUB:
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, "images.txt"))
        label_txt_file = open(os.path.join(self.root, "image_class_labels.txt"))
        train_val_file = open(os.path.join(self.root, "train_test_split.txt"))
        box_file = open(
            os.path.join("/data0/yy_data/Dataset/CUB_1", "bounding_boxes.txt")
        )
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(" ")[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(" ")[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(" ")[-1]))
        box_file_list = []
        for line in box_file:
            # id <x> <y> <width> <height>
            data = line[:-1].split(" ")
            box_file_list.append(
                [
                    int(float(data[2])),  # y
                    int(float(data[1])),  # x
                    int(float(data[4])),  # height
                    int(float(data[3])),  # width
                ]
            )
        # 根据列表中id是否存在来区分训练和测试
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.train_box = torch.tensor(
            [x for i, x in zip(train_test_list, box_file_list) if i]
        )
        self.test_box = torch.tensor(
            [x for i, x in zip(train_test_list, box_file_list) if not i]
        )
        if self.is_train:
            self.train_img = [
                os.path.join(self.root, "images", train_file)
                for train_file in train_file_list[:data_len]
            ]

            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][
                :data_len
            ]
        if not self.is_train:
            self.test_img = [
                os.path.join(self.root, "images", test_file)
                for test_file in test_file_list[:data_len]
            ]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][
                :data_len
            ]

    def __getitem__(self, index):
        if self.is_train:
            img, target, box = (
                imageio.imread(self.train_img[index]),
                self.train_label[index],
                self.train_box[index],
            )
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode="RGB")

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize(
                (self.input_size, self.input_size),
                interpolation=InterpolationMode.BICUBIC,
            )(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                img
            )

        else:
            img, target, box = (
                imageio.imread(self.test_img[index]),
                self.test_label[index],
                self.test_box[index],
            )
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode="RGB")

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize(
                (self.input_size, self.input_size),
                interpolation=InterpolationMode.BICUBIC,
            )(img)
            # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(448)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                img
            )

        scale = torch.tensor([height_scale, width_scale])

        return img, target, box, scale
        # return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class CUB_self:
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        test_file = open(os.path.join(self.root, "test.txt"))
        train_file = open(os.path.join(self.root, "train.txt"))
        box_file = open(
            os.path.join("/data0/yy_data/Dataset/CUB_1", "bounding_boxes.txt")
        )
        train_val_file = open(os.path.join(self.root, "train_test_split.txt"))
        self.train_path = []
        self.test_path = []
        box_file_list = []
        train_test_list = []
        self.train_label = []
        self.test_label = []
        # 获取训练\验证标记
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(" ")[-1]))
        for line in box_file:
            data = line[:-1].split(" ")
            box_file_list.append(
                [  # id <x> <y> <width> <height>
                    int(float(data[2])),
                    int(float(data[1])),
                    int(float(data[4])),
                    int(float(data[3])),
                ]
            )
        self.train_box = torch.tensor(
            [x for i, x in zip(train_test_list, box_file_list) if i]
        )
        self.test_box = torch.tensor(
            [x for i, x in zip(train_test_list, box_file_list) if not i]
        )
        # 训练
        if self.is_train:
            for line in train_file:
                self.train_path.append(
                    [line[:-1].split(" ")[0], int(line[:-1].split(" ")[1])]
                )
                self.train_label.append([int(line[:-1].split(" ")[1])])
        # 验证
        if not self.is_train:
            for line in test_file:
                self.test_path.append(
                    [line[:-1].split(" ")[0], int(line[:-1].split(" ")[1])]
                )
                self.test_label.append([int(line[:-1].split(" ")[1])])

    def __getitem__(self, index):
        if self.is_train:
            img, target, box = (
                imageio.imread(self.train_path[index][0]),
                self.train_label[index],
                self.train_box[index],
            )

            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode="RGB")
            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width
            img = transforms.Resize(
                (self.input_size, self.input_size),
                interpolation=InterpolationMode.BICUBIC,
            )(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                img
            )

        else:
            img, target, box = (
                imageio.imread(self.test_path[index][0]),
                self.test_label[index],
                self.test_box[index],
            )
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode="RGB")
            # compute scale
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width
            img = transforms.Resize(
                (self.input_size, self.input_size),
                interpolation=InterpolationMode.BICUBIC,
            )(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                img
            )
        scale = torch.tensor([height_scale, width_scale])
        # return img, target, box, scale
        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


# 读取数据集
def read_dataset(input_size, batch_size, root, set):
    if set == "CUB":
        print("Loading CUB trainset")
        trainset = CUB(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )
        print("Loading CUB testset")
        testset = CUB(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )
    elif set == "Planet":
        print("Loading Planet trainset")
        # trainset = dataset.Inaturalist_21_plante(
        #     input_size=input_size, root=root, is_train=True
        # )
        # trainloader = torch.utils.data.DataLoader(
        #     trainset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=8,
        #     drop_last=False,
        # )
        print("Loading Planet testset")
        # testset = dataset.Inaturalist_21_plante(
        #     input_size=input_size, root=root, is_train=False
        # )
        # testloader = torch.utils.data.DataLoader(
        #     testset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=8,
        #     drop_last=False,
        # )
    else:
        print("Please choose supported dataset")
        os._exit(1)

    return trainloader, testloader


if __name__ == "__main__":
    # root = "/data0/yy_data/Dataset/CUB_1/"
    # dataset = CUB_self(448, root)
    # print("dataset data:", dataset[0][0])
    # print("dataset label:", dataset[0][1])
    train_loader, test_loader = read_dataset(
        448, 6, "/data0/yy_data/Dataset/CUB_1/", "CUB"
    )
    print("len train {}".format(len(train_loader)))
    print("len test {}".format(len(test_loader)))
