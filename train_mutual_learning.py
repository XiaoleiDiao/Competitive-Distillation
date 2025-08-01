import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import csv
import wandb
import argparse
import torchvision
import numpy as np
import utils
import random
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def parse_args():
    parser = argparse.ArgumentParser()

    '''
    Model Selections：resnet20, resnet32, resnet44, resnet56, resnet152, resnet1202. mobilenetV1, inceptionV1
    '''
    parser.add_argument("--model", type=str, default="resnet32", required=False)
    parser.add_argument("--dataset", type=str, default='cifar100', required=False)
    parser.add_argument("--gpu", type=str, default="0,1", required=False)
    parser.add_argument("--num_classes", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--num_workers", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=400, required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--optim", type=str, default='adamw', required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--input_size", type=int, default=224, required=False)
    parser.add_argument("--lrscheduler", type=str, default='reduce', required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0005, required=False)
    parser.add_argument("--momentum", type=float, default=0.9, required=False)
    parser.add_argument("--gamma", type=float, default=0.1, required=False)
    parser.add_argument("--device", type=str, default='cuda', required=False)
    parser.add_argument("--topk", type=int, default=1, required=False)
    parser.add_argument("--eval_flag", type=bool, default=True, required=False)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    return args

Augmentation = torchvision.transforms.Compose(
        [
            torchvision.transforms.ColorJitter(
                brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15
            ),
            torchvision.transforms.GaussianBlur(kernel_size=(7, 9)),
        ]
    )
Random_Rotation = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomRotation(30)
    ]
)
# Perturbation pool: There are many different perturbation methods.
Disturb_arr = [
    'mixup', 'guassian_noise', 'color_jitter_and_guassian_blur', 'hfilp', 'vflip', 'random_rotation'
]

# define a perturbation pool
def get_disturb_return(images, labels):
    m_length = len(Disturb_arr)
    random_idx = random.randint(0, m_length - 1)
    method_name = Disturb_arr[random_idx]
    lam = 0
    
    if method_name == 'mixup':
        re_images, re_labels, lam = utils.mixed_image(images, labels)
        
    elif method_name == 'guassian_noise':
        noise = np.random.normal(0, 1, images.shape)
        noise = torch.from_numpy(noise).cuda()
        re_images = images + noise
        re_images = torch.clamp(re_images, 0, 1)
        re_labels = labels
        
    elif method_name == 'color_jitter_and_guassian_blur':
        re_images = Augmentation(images)
        re_labels = labels
        
    elif method_name == 'hfilp':
        re_images = TF.hflip(images)
        re_labels = labels
        
    elif method_name == 'vflip':
        re_images = TF.vflip(images)
        re_labels = labels
        
    elif method_name == 'random_rotation':
        re_images = Random_Rotation(images)
        re_labels = labels
        
    else:
        raise ValueError('disturb method is not define, please define it...')
    return re_images, re_labels, lam


def train_one_epoch(epoch,
                    models,
                    train_dataloader,
                    optimizers,
                    criterion_ce,
                    criterion_kl,
                    train_losses,
                    args,
                    is_augmentation=False
                    ):

    with tqdm(total=len(train_dataloader), desc=f'{epoch}/{args.epochs}', postfix=dict, maxinterval=1.0) as pbar:

        for i_batch, (images, labels) in enumerate(train_dataloader):
            # Whether to perform perturbation:
            images = images.cuda()
            labels = labels.cuda()
            if i_batch % 10 == 0:
                re_images, re_labels, lam = get_disturb_return(images, labels)
            else:
                re_images = images
                re_labels = labels
                lam = 0
            re_images = re_images.cuda()
            re_labels = re_labels.cuda()

            images = images.to(args.device)
            labels = labels.to(args.device)

            # Calculate the output of each model:
            random_idx = np.random.choice(len(models))
            outputs = utils.compute_models_output(models, images, optimizers, re_images, random_idx)

            # Calculate the loss of each model:
            ce_losses = [criterion_ce(outputs[i], labels) for i in range(len(models))]
            if lam > 0:
                ce_losses[random_idx] = lam * F.cross_entropy(outputs[random_idx], labels) \
                                        + (1 - lam) * F.cross_entropy(outputs[random_idx], re_labels)

            for i in range(len(models)):
                ce_loss = ce_losses[i]
                kl_loss = 0.

                for j in range(len(models)):
                    if i != j:
                        kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(outputs[j].detach(), dim=1))

                loss = ce_loss + kl_loss / (len(models) - 1)

                # measure current metric
                _, predict = outputs[i].topk(args.topk, 1, True, True)
                predict = predict.t()
                acc = predict.eq(labels.view_as(predict)).float().sum()
                if lam>0 and random_idx == i:
                    acc = (lam * predict.eq(labels.data).cpu().sum().float()
                            + (1 - lam) * predict.eq(re_labels.data).cpu().sum().float())

                loss.backward()
                optimizers[i].step()

                # record the loss
                train_losses[i].update(ce_loss.item(), kl_loss.item(), loss.item(), acc.item(), 0,epoch)

                # measure the loss in current step
                pbar.set_postfix({
                    # 'model': i,
                    # 'ce_loss': train_losses[i].ce_loss / (i_batch + 1),
                    # 'kl_loss': train_losses[i].kl_loss / (i_batch + 1),
                    'loss': train_losses[i].loss / (i_batch + 1),
                    'correct': train_losses[i].acc / len(train_dataloader.dataset),
                })

            pbar.update(1)

@torch.no_grad()
def val(epoch, models, val_dataloader, criterion_ce, criterion_kl, val_losses, args):

    for i in range(len(models)):
        models[i].eval()

    with tqdm(total=len(val_dataloader), desc=f'Epoch [{epoch}/{args.epochs}]') as pbar:
        for i_batch, (images, labels) in enumerate(val_dataloader):
            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = utils.compute_models_output(models, images)

            ce_losses = [criterion_ce(outputs[i], labels) for i in range(len(models))]

            # calculate the loss of each model
            for i in range(len(models)):
                ce_loss = ce_losses[i]
                kl_loss = 0.

                for j in range(len(models)):
                    if i != j:
                        kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(outputs[j].detach(), dim=1))

                loss = ce_loss + kl_loss / (len(models) - 1)

                # measure current metric
                _, predict = outputs[i].topk(args.topk, 1, True, True)
                predict = predict.t()
                acc = predict.eq(labels.view_as(predict)).float().sum()

                # record the loss of model
                val_losses[i].update(ce_loss.item(), kl_loss.item(), loss.item(), acc.item(),0, epoch)
                pbar.set_postfix({
                    "acc": val_losses[i].acc / len(val_dataloader.dataset),
                })

            pbar.update(1)


def train(args, model_str, project_name_, SAVE_PATH, is_augmentation):
    utils.set_random_seed(args.seed)
    args.model = model_str
    for i in range(10):
        try:
             wandb.init(project='mutual_involution_learning', name=project_name_)
             break
        except:
            print('wandb init failed')
            continue


    wandb.config.update(args, allow_val_change=True)

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')

    model_names, \
    models, \
    train_dataloader, \
    val_dataloader = utils.get_model_and_dataset(args)

    optimizers = utils.get_optimizer(args, models, len(models))
    schedulers = utils.get_scheduler(args, optimizers)

    n_train_batch = len(train_dataloader)
    n_train_data = len(train_dataloader.dataset)
    n_val_batch = len(val_dataloader)
    n_val_data = len(val_dataloader.dataset)
    best_val_acc = [0.] * len(models)
    best_train_acc = [0.] * len(models)

    utils.show_config(args)

    for epoch in range(args.epochs):
        train_losses = []
        val_losses = []

        for i in range(len(models)):
            train_losses.append(utils.Totoal_Meter(n_train_batch, n_train_data))
            val_losses.append(utils.Totoal_Meter(n_val_batch, n_val_data))
            schedulers[i].step(epoch)
            models[i] = models[i].cuda().train()

        # train
        train_one_epoch(epoch, models, train_dataloader, optimizers,
                              criterion_ce, criterion_kl, train_losses, args, is_augmentation)

        # val
        if args.eval_flag:
            val(epoch, models, val_dataloader,
                      criterion_ce, criterion_kl, val_losses, args)

        # save and print
        for i in range(len(models)):
            # lr = utils.get_lr(optimizers[i])
            train_res = train_losses[i].get_avg()
            val_res = val_losses[i].get_avg()

            if val_res['acc'] > best_val_acc[i]:
                best_val_acc[i] = val_res['acc']
            if train_res['acc'] > best_train_acc[i]:
                best_train_acc[i] = train_res['acc']
                # torch.save(models[i].state_dict(), os.path.join(SAVE_PATH, f'{i}_{model_names[i]}_best.pth'))

            # save training information inwandb
            wandb.log({
                "{}_{}_{}_{}_ceLoss".format(i, model_names[i], args.dataset, 'train'): train_res['ce_loss'],
                "{}_{}_{}_{}_klLoss".format(i, model_names[i], args.dataset, 'train'): train_res['kl_loss'],
                "{}_{}_{}_{}_loss".format(i, model_names[i], args.dataset, 'train'): train_res['loss'],
                "{}_{}_{}_{}_acc".format(i, model_names[i], args.dataset, 'train'): train_res['acc'],
                "{}_{}_{}_{}_loss".format(i, model_names[i], args.dataset, 'val'): val_res['loss'],
                "{}_{}_{}_{}_acc".format(i, model_names[i], args.dataset, 'val'): val_res['acc'],
            },step=epoch)

        with open(os.path.join(SAVE_PATH, 'mutual_result.csv'), 'w', encoding='utf-8', newline='') as f:
            fileName = ['model_name', 'train_acc', 'val_acc']
            writer = csv.DictWriter(f, fieldnames=fileName)
            writer.writeheader()
            for i in range(len(model_names)):
                writer.writerow({'model_name':model_names[i],
                                 'train_acc': best_train_acc[i],
                                 'val_acc': best_val_acc[i],})

    wandb.finish()


def mutual_(model_str, proj_name, SAVE_PATH, is_augmentation):
    args = parse_args()
    train(args, model_str, proj_name, SAVE_PATH, is_augmentation=is_augmentation)

if __name__ == '__main__':
    mutual_()




