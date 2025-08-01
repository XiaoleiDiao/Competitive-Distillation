import glob

from tqdm import tqdm
import torch.nn.functional as F
import torch
import os
import random
import numpy as np
from torch.optim import *
import models.resnet as resnet
import models.mobilenet as mobilenet
import models.inception as inception
import datasets.dataset as dataset
from models.CeiT import CeiT
from models.vit import ViT

class Totoal_Meter(object):
    def __init__(self, n_batch, n_dataset):
        self.n_batch = n_batch
        self.n_dataset = n_dataset
        self.reset()

    def reset(self):
        self.ce_loss = 0.
        self.kl_loss = 0.
        self.loss = 0.
        self.acc = 0.
        self.epoch = 0
        self.l2_loss = 0.

    def update(self, ce_loss=0, kl_loss=0, loss=0, acc=0, l2_loss=0, epoch=0):
        self.ce_loss += ce_loss
        self.kl_loss += kl_loss
        self.loss += loss
        self.acc += acc
        self.epoch = epoch
        self.l2_loss = l2_loss

    def get_avg(self):
         return {
            'ce_loss': self.ce_loss / self.n_batch,
            'kl_loss': self.kl_loss / self.n_batch,
            'loss': self.loss / self.n_batch,
            'acc': self.acc / self.n_dataset,
            'l2_loss': self.l2_loss / self.n_batch,
            'epoch': self.epoch,
        }

def show_config(args):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)

    for key, value in vars(args).items():
        print('|%25s | %40s|' % (str(key), str(value)))

    print('-' * 70)


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_random_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_scheduler(args, optimizers):
    schedulers = []
    for i in range(len(optimizers)):
        if args.lrscheduler == 'step':
            scheduler = lr_scheduler.StepLR(optimizers[i],
                                            step_size=50,
                                            gamma=args.gamma,
                                            last_epoch=-1)
        elif args.lrscheduler == 'reduce':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizers[i],
                                                       mode='max',
                                                       factor=0.1,
                                                       patience=5,
                                                       verbose=True,
                                                       min_lr=0.000005)
        else:
            raise ValueError('lrscheduler is not specified')

        schedulers.append(scheduler)
    return schedulers


def get_optimizer(args, models, n_model):
    optimizers = []
    for i in range(n_model):

        if args.optim == 'adam':
            optimizer = torch.optim.Adam(models[i].parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)  # ,weight_decay=0.1
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(models[i].parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)

        elif args.optim == 'adamw':
            optimizer_parameters = filter(lambda p: p.requires_grad, models[i].parameters())
            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-8,
                # correct_bias=False
            )

        optimizers.append(optimizer)

    return optimizers


def get_model_and_dataset(args):
    model_names = [m.strip() for m in args.model.split(',')]
    dataset_name = args.dataset

    # load all models
    models = []
    for m in model_names:
        print('m = ',m)
        if m.find('resnet') != -1 or m == 'wideResnet28_12' :
            model = getattr(resnet, m)(num_classes=args.num_classes)
        elif m.find('mobilenet') != -1:
            model = getattr(mobilenet, m)(num_classes=args.num_classes)
        elif m.find('inception') != -1:
            model = getattr(inception, m)(num_classes=args.num_classes)
        elif m.find('ceit') != -1:
            model = CeiT(args.input_size, 4, args.num_classes)
        elif m.find('ViT') != -1:
            model = ViT(args.input_size, args.num_classes)
        else:
            raise ValueError('do not have this model now')
        models.append(model)

    # load datasets
    train_dataloader, \
    test_dataloader = getattr(dataset, dataset_name)(args)

    return model_names, models, train_dataloader, test_dataloader


def get_save_path():
    paths = glob.glob(os.path.join('result', 'runs*'))
    if len(paths) == 0:
        SAVE_PATH = r'result' + os.sep + 'runs1'
        os.mkdir(SAVE_PATH)
    else:
        ids = sorted([int(p.replace('result' + os.sep + 'runs', '')) for p in paths], reverse=True)
        print(ids)
        ss = str(ids[0] + 1)
        SAVE_PATH = f'result' + os.sep + f'runs{ss}'
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
    return SAVE_PATH


def mixed_image(imgs, labels):
    alpha = 1.0
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    index = torch.randperm(imgs.shape[0])
    imgs_A, imgs_B = imgs, imgs[index]
    labels_A, labels_B = labels, labels[index]

    # mixup
    new_imgs = lam * imgs_A + (1 - lam) * imgs_B

    return new_imgs, labels_B, lam


def compute_models_output(models, images, optimizers=None, mixed_imgs=None, random_idx=None):
    outputs = []
    if optimizers is not None:
        # train
        for i, model in enumerate(models):
            optimizers[i].zero_grad()
            if isinstance(model, CeiT):
                mixed_imgs = mixed_imgs.float()
                final_predict, _ = model(mixed_imgs) if i == random_idx else model(images)
            else:
                mixed_imgs = mixed_imgs.float()
                final_predict = model(mixed_imgs) if i == random_idx else model(images)
            outputs.append(final_predict)
    else:
        # val
        for i, model in enumerate(models):
            if isinstance(model, CeiT):
                final_predict, _ = model(images)
            else:
                final_predict = model(images)
            outputs.append(final_predict)

    return outputs

def compute_models_output_invo(models, images, optimizers=None, mixed_imgs=None, random_idx=None):
    outputs = []
    final_features = []

    if optimizers is not None:
        # train
        for i, model in enumerate(models):
            optimizers[i].zero_grad()
            if isinstance(model, CeiT):
                mixed_imgs = mixed_imgs.float()
                final_predict, final_feature = model(mixed_imgs) if i == random_idx else model(images)
                final_features.append(final_feature)
            else:
                mixed_imgs = mixed_imgs.float()
                final_predict = model(mixed_imgs) if i == random_idx else model(images)
            outputs.append(final_predict)
    else:
        # val
        for i, model in enumerate(models):
            if isinstance(model, CeiT):
                final_predict, final_feature = model(images)
                final_features.append(final_feature)
            else:
                final_predict = model(images)
            outputs.append(final_predict)

    return outputs, final_features

if __name__ == '__main__':
    model = CeiT(224,4,100) # Total params: 5578852
    import torchsummary
    torchsummary.summary(model, (3,224,224), 100, device='cpu')

    model = ViT(224, 100) # Total params: 85705060
    torchsummary.summary(model, (3, 224, 224), 100, device='cpu')

    model = resnet.resnet32(100) # Total params: 470004
    torchsummary.summary(model, (3, 224, 224), 100, device='cpu')

    model = mobilenet.mobilenetV1(100)  # Total params: 470004
    torchsummary.summary(model, (3, 224, 224), 100, device='cpu')
