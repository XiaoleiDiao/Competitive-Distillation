import argparse
import os
import time

import PIL.Image
import numpy as np
import torch
import sys
import torchvision
import torchvision.transforms as transforms


# CIFAR10 dataset
def cifar10(args):
    CIFAR_PATH = 'data/cifar10'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # transfer images to 32 * 32
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=CIFAR_PATH,
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR10(root=CIFAR_PATH,
                                                train=False,
                                                download=True,
                                                transform=transform_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

    return train_dataloader, test_dataloader


# CIFAR100 dataset
def cifar100(args):
    CIFAR_PATH = "data/cifar100/"

    transform_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),  # transfer images to 32 * 32
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.CIFAR100(root=CIFAR_PATH,
                                                  train=True,
                                                  download=True,
                                                  transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR100(root=CIFAR_PATH,
                                                 train=False,
                                                 download=True,
                                                 transform=transform_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

    return train_dataloader, test_dataloader


# Places365 dataset
def Places365(args):
    Places365_path ='data/place365/'

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # transfer images to 32 * 32
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.Places365(root=Places365_path,
                                                        split='train-standard',
                                                        small=True,
                                                        transform=transform_train)

    test_dataset = torchvision.datasets.Places365(root=Places365_path,
                                                       split='val',
                                                       small=True,
                                                       transform=transform_test)

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

    return trainloader, testloader


def imagenet(args):
    traindir = 'data/imagenet/train/traindata/'
    traindir = r'data/imagenet2012/train'
    valdir = 'data/imagenet/val/'#os.path.join(root, 'ILSVRC2012_img_val')
    valdir = r'data/imagenet2012/val'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    print('trainSet num: ',len(train_dataset))

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )
    print('valSet num: ', len(val_dataset))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=None
    )
    print('train train_dataloader: ', len(train_dataloader))

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    print('train val_dataloader: ', len(val_dataloader))

    return train_dataloader, val_dataloader


# image_tiny dataset
def imagenet_tiny(args):
    traindir = 'data/imagenet_tiny/tiny-imagenet-200/val/'
    valdir = 'data/imagenet_tiny/tiny-imagenet-200/val/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


def cub_200(args):
    root_train = 'data/cub_200/CUB_200_2011/dataset/train/'
    root_test = 'data/cub_200/CUB_200_2011/dataset/test/'  # os.path.join(root, 'ILSVRC2012_img_val')
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

    # training set
    train_dataset = torchvision.datasets.ImageFolder(root_train, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    # test set
    test_dataset = torchvision.datasets.ImageFolder(root_test, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    return train_loader, test_loader


def market(args):
    if args.use_swin:
        h, w = 224, 224
    else:
        h, w = 256, 128#160, 64

    image_datasets = {}
    data_dir ='data/Market-1501-v15.09.15/Market-1501/pytorch/'
    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=(h, w), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    image_datasets['train'] = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                                   data_transforms['train'])
    image_datasets['val'] = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                                 data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=8)  # 8 workers may work faster
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders['train'], dataloaders['val']


def market_dataset_test(args):
    if args.use_swin:
        h, w = 224, 224
    else:
        h, w = 160, 64
    image_datasets = {}
    data_dir ='data/Market-1501-v15.09.15/Market-1501/pytorch/'
    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # if args.type == 'multi':
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                      ['gallery', 'query', 'multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=False, num_workers=16) for x in
                   ['gallery', 'query', 'multi-query']}
    # if args.type == 'single':
    #     image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
    #                       ['gallery', 'query', 'multi-query']}
    #     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
    #                                                   shuffle=False, num_workers=16) for x in
    #                    ['gallery', 'query', 'multi-query']}
    return image_datasets, dataloaders


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.batch_size = 512
    args.num_workers = 8
    imagenet(args)
    exit()
    train,test = cifar100()
    train = train.dataset
    test = test.dataset
    print(len(train), len(test))
    print(len(train[0]))
    img = train[0][0]
    label = train[0][1]
    print(type(img), type(label))
    print(img.shape)
    print(img)
    img = img * 255
    img = img.numpy().astype(np.uint32)
    print(img)
    img = PIL.Image.fromarray(img.numpy().astype(np.uint8), 'RGB')
    img.show()
    print(label)
