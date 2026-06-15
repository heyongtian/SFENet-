
import os
import PIL

import os, random, glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

random.seed(42)

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


## pretrain
class SeismicSet(data.Dataset):

    def __init__(self, path, input_size) -> None:
        super().__init__()
        # self.file_list = os.listdir(path)
        # self.file_list = [os.path.join(path, f) for f in self.file_list]
        self.get_file_list(path)
        self.input_size = input_size
        print(len(self.file_list))

    def __len__(self) -> int:
        return len(self.file_list)
        # return 100000

    def __getitem__(self, index):
        d = np.fromfile(self.file_list[index], dtype=np.float32)
        d = d.reshape(1, self.input_size, self.input_size)
        d = (d - d.mean()) / (d.std() + 1e-6)

        # return to_transforms(d, self.input_size)
        return d, torch.tensor([1])

    def get_file_list(self, path):
        dirs = [os.path.join(path, f) for f in os.listdir(path)]
        self.file_list = dirs

        # for ds in dirs:
        #     if os.path.isdir(ds):
        #         self.file_list += [os.path.join(ds, f) for f in os.listdir(ds)]

        return random.shuffle(self.file_list)


def to_transforms(d, input_size):
    t = transforms.Compose([
        transforms.RandomResizedCrop(input_size,
                                     scale=(0.2, 1.0),
                                     interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    return t(d)

class DenoiseSet(data.Dataset):
    def __init__(self,
                 folder,
                 shape=[224, 224],
                 is_train=True) -> None:
        super().__init__()
        self.shape = shape
        self.data_list = [folder + 'seismic/' + str(f) + '.dat' for f in range(2000)]
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list
            self.label_list = [f.replace('/seismic/', '/label/') for f in self.data_list]
        elif not is_train:
            self.data_list = [folder + 'feature/' + str(f) + '.dat' for f in range(4000)]
            self.label_list = self.data_list

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        # d = (d - d.mean())/d.std()
        l = np.fromfile(self.label_list[index], np.float32)
        l = l.reshape([1] + self.shape)
        # l = (l - d.mean())/l.std()
        # d为feature，l为label
        return torch.tensor(d), torch.tensor(l)

    def __len__(self):
        return len(self.data_list)


