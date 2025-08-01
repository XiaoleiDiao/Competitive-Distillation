import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.optim import lr_scheduler
import torch
import datasets.dataset
import wandb
import argparse
from models.CeiT import CeiT
import utils
from tqdm import tqdm
import csv
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    '''
    Model selections：resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202. mobilenetV1, inceptionV1
    '''
    parser.add_argument("--model", type=str, default="ceit", required=False)
    parser.add_argument("--input_size", type=int, default=224, required=False)
    parser.add_argument("--dataset", type=str, default='cifar100', required=False)
    parser.add_argument("--gpu", type=str, default="1", required=False)
    parser.add_argument("--num_classes", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=96, required=False)
    parser.add_argument("--num_workers", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=200, required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--optim", type=str, default='adamw', required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--lrscheduler", type=str, default='reduce', required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0005, required=False)
    parser.add_argument("--momentum", type=float, default=0.9, required=False)
    parser.add_argument("--gamma", type=float, default=0.1, required=False)
    parser.add_argument("--device", type=str, default='cuda', required=False)
    parser.add_argument("--topk", type=int, default=1, required=False)
    parser.add_argument("--eval_flag", type=bool, default=True, required=False)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    return args


def train_one_epoch(epoch,
                    model,
                    train_dataloader,
                    optimizer,
                    criterion_ce,
                    train_losses,
                    args,
                    is_augmentation=True
                    ):

    with tqdm(total=len(train_dataloader), desc=f'{epoch}/{args.epochs}', postfix=dict, maxinterval=1.0) as pbar:

        for i_batch, (images, labels) in enumerate(train_dataloader):
            images = images.cuda()
            labels = labels.cuda()
            if is_augmentation:
                mixed_imgs, mixed_labels, lam = utils.mixed_image(images, labels)
                mixed_imgs = mixed_imgs.cuda()
                mixed_labels = mixed_labels.cuda()

            optimizer.zero_grad()
            output, _ = model(mixed_imgs) if is_augmentation else model(images)

            if is_augmentation:
                loss = lam * criterion_ce(output, labels) + (1 - lam) * criterion_ce(output, mixed_labels)
            else:
                loss = criterion_ce(output, labels)

            # Measure the accuracy currently
            _, predict = output.topk(args.topk, 1, True, True)
            predict = predict.t()

            if is_augmentation:
                acc = (lam * predict.eq(labels.data).cpu().sum().float()
                       + (1 - lam) * predict.eq(mixed_labels.data).cpu().sum().float())
            else:
                acc = predict.eq(labels.view_as(predict)).float().sum()

            loss.backward()
            optimizer.step()

            train_losses.update(loss=loss.item(), acc=acc.item(), epoch=epoch)
            pbar.set_postfix({
                'loss': train_losses.loss / (i_batch + 1),
                'acc': train_losses.acc / len(train_dataloader.dataset),
            })

            pbar.update(1)


def val(epoch, model, val_dataloader, criterion_ce, val_losses, args):
    with torch.no_grad():
        model.eval()

        with tqdm(total=len(val_dataloader), desc=f'Epoch [{epoch}/{args.epochs}]') as pbar:
            for i_batch, (images, labels) in enumerate(val_dataloader):

                images = images.cuda()
                labels = labels.cuda()
                output, _ = model(images)

                loss = criterion_ce(output, labels)

                _, predict = output.topk(args.topk, 1, True, True)
                predict = predict.t()
                acc = predict.eq(labels.view_as(predict)).float().sum()

                val_losses.update(loss=loss.item(), acc=acc.item(), epoch=epoch)
                pbar.set_postfix({
                    'correct': val_losses.acc / len(val_dataloader.dataset),
                })

                pbar.update(1)


def train(args, project_name, SAVE_PATH):
    utils.set_random_seed(args.seed)

    wandb.init(project='mutual_involution_learning', name=project_name)
    wandb.config.update(args, allow_val_change=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    criterion_ce = torch.nn.CrossEntropyLoss()

    model = CeiT(args.input_size, 4, args.num_classes)
    train_dataloader, \
    val_dataloader = getattr(datasets.dataset, args.dataset)(args)

    optimizer_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-8,
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode='max',
                                               factor=0.1,
                                               patience=5,
                                               verbose=True,
                                               min_lr=0.000005)

    n_train_batch = len(train_dataloader)
    n_train_data = len(train_dataloader.dataset)
    n_val_batch = len(val_dataloader)
    n_val_data = len(val_dataloader.dataset)
    best_val_acc = 0.
    best_train_acc = 0.

    utils.show_config(args)

    for epoch in range(args.epochs):
        train_losses = utils.Totoal_Meter(n_train_batch, n_train_data)
        val_losses = utils.Totoal_Meter(n_val_batch, n_val_data)

        scheduler.step(epoch)
        model = model.cuda().train()

        # train
        train_one_epoch(epoch, model, train_dataloader, optimizer,
                              criterion_ce, train_losses, args)

        # val
        if args.eval_flag:
            val(epoch, model, val_dataloader,
                      criterion_ce, val_losses, args)

        # save and print
        train_res = train_losses.get_avg()
        val_res = val_losses.get_avg()

        if val_res['acc'] > best_val_acc:
            best_val_acc = val_res['acc']
        if train_res['acc'] > best_train_acc:
            best_train_acc = train_res['acc']
            # torch.save(model.state_dict(), os.path.join('result', f'ceit_best.pth'))

        # save training information in wandb
        wandb.log({
            "{}_{}_{}_loss".format('ceit', args.dataset, 'train'): train_res['loss'],
            "{}_{}_{}_acc".format('ceit', args.dataset, 'train'): train_res['acc'],
            "{}_{}_{}_loss".format('ceit', args.dataset, 'val'): val_res['loss'],
            "{}_{}_{}_acc".format('ceit', args.dataset, 'val'): val_res['acc'],
        }, step=epoch)

    with open(os.path.join(SAVE_PATH, 'result.csv'), 'w', encoding='utf-8', newline='') as f:
        fileName = ['model_name', 'train_acc', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fileName)
        writer.writeheader()
        writer.writerow({'model_name': 'ceit',
                         'train_acc': best_train_acc,
                         'val_acc': best_val_acc})

    wandb.finish()

def main(proj_name, SAVE_PATH):
    args = parse_args()
    train(args, proj_name, SAVE_PATH)

if __name__ == '__main__':
    SAVE_PATH = utils.get_save_path()
    print('Current step SAVE_PATH', SAVE_PATH)
    main('mutual_single_ceit+mixup', SAVE_PATH)




