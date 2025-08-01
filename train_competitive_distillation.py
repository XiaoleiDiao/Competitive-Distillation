import os
import torch
import wandb
import argparse
import random
import utils
from tqdm import tqdm
import torch.nn.functional as F
import csv
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.transforms.functional as TF

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
    
        
def parse_args():
    parser = argparse.ArgumentParser()

    '''
    Model Selection：resnet20, resnet32, resnet44, resnet56, resnet152, resnet1202. mobilenetV1, inceptionV1
    '''
    parser.add_argument("--model", type=str, default="resnet32", required=False)
    parser.add_argument("--input_size", type=int, default=224, required=False)
    parser.add_argument("--dataset", type=str, default='cifar100', required=False)
    parser.add_argument("--gpu", type=str, default="0,1", required=False)
    parser.add_argument("--num_classes", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=84, required=False)
    parser.add_argument("--num_workers", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=400, required=False)
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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    return args


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
            # Randomly select a perturbation method for training
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
            
            # Randomly select a model in each step
            random_idx = np.random.choice(len(models))
            
            # Calculate the output of each model:
            outputs, final_features = utils.compute_models_output_invo(models, images, optimizers, re_images, random_idx)

            # Calculate the loss of each model:
            ce_losses = [criterion_ce(outputs[i], labels) for i in range(len(models))]
            if lam > 0:
                ce_losses[random_idx] = lam * criterion_ce(outputs[random_idx], labels) \
                                            + (1 - lam) * criterion_ce(outputs[random_idx], re_labels)

            # ----------------------
            # 1. competitive distillation, learn for the best-performing model
            # ----------------------
            for i in range(len(models)):
                # Calculate the kl_loss of each model, if the ce_loss of the model is larger than the ce_loss of the traversed model,
                # then it will be calculated in kl_loss
                ce_loss = ce_losses[i]
                l2_loss = 0.
                kl_loss = 0.
                n_kl = 0. # Record the number of competitive distillation learning models, used for averaging
                lr = utils.get_lr(optimizers[i])

               # 1. Compare models with each other, and learn from all models that perform better than themselves
                for j in range(len(models)):
                    if i != j and ce_loss > ce_losses[j]:
                        n_kl += 1
                        kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(outputs[j].detach(), dim=1))
                        # l2_loss += F.mse_loss(final_features[i], Variable(final_features[j]))

                # full Loss: cross-entropy + KLdiv + L2-loss
                # loss = ce_loss + (0.1 * l2_loss + kl_loss) / n_kl if n_kl != 0 else ce_loss
                loss = ce_loss + (kl_loss) / n_kl if n_kl != 0 else ce_loss

                # measure current metric
                _, predict = outputs[i].topk(args.topk, 1, True, True)
                predict = predict.t()

                acc = predict.eq(labels.view_as(predict)).float().sum()
                if lam > 0 and i == random_idx:
                    acc = (lam * predict.eq(labels.data).cpu().sum().float()
                           + (1 - lam) * predict.eq(re_labels.data).cpu().sum().float())

                loss.backward()
                optimizers[i].step()

                # save the loss of the model
                train_losses[i].update(ce_loss.item(),
                                       kl_loss.item() if kl_loss!=0. else kl_loss,
                                       loss.item(),
                                       acc.item(),
                                       l2_loss.item() if l2_loss!=0. else l2_loss,
                                       epoch)

                # Show the loss for the current training epoch
                pbar.set_postfix({
                    # 'model': i,
                    # 'ce_loss': train_losses[i].ce_loss / (i_batch + 1),
                    # 'kl_loss': train_losses[i].kl_loss / (i_batch + 1),
                    'loss': train_losses[i].loss / (i_batch + 1),
                    'correct': train_losses[i].acc / len(train_dataloader.dataset),
                    'lr': lr,
                })

            pbar.update(1)


def val(epoch, models, val_dataloader, criterion_ce, criterion_kl, val_losses, args):
    with torch.no_grad():
        for i in range(len(models)):
            models[i].eval()

        with tqdm(total=len(val_dataloader), desc=f'Epoch [{epoch}/{args.epochs}]') as pbar:
            for i_batch, (images, labels) in enumerate(val_dataloader):

                images = images.cuda()
                labels = labels.cuda()

                # calculate the output of each model
                outputs, final_features = utils.compute_models_output_invo(models, images)

                ce_losses = [criterion_ce(outputs[i], labels) for i in range(len(models))]

                # ----------------------
                # 1.
                # ----------------------
                # calculate the loss of each model
                for i in range(len(models)):
                    ce_loss = ce_losses[i]
                    l2_loss = 0.
                    kl_loss = 0.
                    n_kl = 0

                    for j in range(len(models)):
                        if i != j and ce_loss > ce_losses[j]:
                            n_kl += 1
                            kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
                                                    F.softmax(Variable(outputs[j]), dim=1))
                            # l2_loss += F.mse_loss(final_features[i], Variable(final_features[j]))

                    # loss = ce_loss + (0.1 * l2_loss + kl_loss) / n_kl if n_kl != 0 else ce_loss
                    loss = ce_loss + (kl_loss) / n_kl if n_kl != 0 else ce_loss

                    # measure current metric
                    _, predict = outputs[i].topk(args.topk, 1, True, True)
                    predict = predict.t()
                    acc = predict.eq(labels.view_as(predict)).float().sum()

                    # record the loss of model
                    val_losses[i].update(0, 0,
                                         loss.item(),
                                         acc.item(),
                                         l2_loss.item() if l2_loss!=0 else l2_loss,
                                         epoch)
                pbar.set_postfix({
                    'correct': val_losses[i].acc / len(val_dataloader.dataset),
                })
                pbar.update(1)


def train(args, model_str, project_name, SAVE_PATH, is_augmentation=False):
    utils.set_random_seed(args.seed)
    args.model = model_str
    wandb.init(project='competitive_distillation', name=project_name)
    wandb.config.update(args, allow_val_change=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')

    model_names, \
    models, \
    train_dataloader, \
    val_dataloader = utils.get_model_and_dataset(args)

    for l in range(len(models)):
        models[l] = models[l].cuda()

    optimizers = utils.get_optimizer(args, models, len(models))
    schedulers = utils.get_scheduler(args, optimizers)

    n_train_batch = len(train_dataloader)
    n_train_data = len(train_dataloader.dataset)
    n_val_batch = len(val_dataloader)
    n_val_data = len(val_dataloader.dataset)
    best_val_acc = [0.] * len(models)
    best_train_acc = [0.] * len(models)

    utils.show_config(args)

    # Multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f'use {torch.cuda.device_count()} GPUs')
        device = torch.device('cuda:0')
        new_models = []
        for k in range(len(models)):
            model0 = torch.nn.DataParallel(models[k], device_ids=[0,1]).cuda()
            new_models.append(model0)
        models = new_models

    for epoch in range(args.epochs):
        train_losses = []
        val_losses = []

        for i in range(len(models)):
            train_losses.append(utils.Totoal_Meter(n_train_batch, n_train_data))
            val_losses.append(utils.Totoal_Meter(n_val_batch, n_val_data))
            schedulers[i].step(epoch)
            models[i] = models[i].train()

        # train
        train_one_epoch(epoch, models, train_dataloader, optimizers,
                              criterion_ce, criterion_kl, train_losses, args, is_augmentation)

        # val
        if args.eval_flag:
            val(epoch, models, val_dataloader,
                      criterion_ce, criterion_kl, val_losses, args)

        # save and print
        for i in range(len(models)):
            train_res = train_losses[i].get_avg()
            val_res = val_losses[i].get_avg()

            if val_res['acc'] > best_val_acc[i]:
                best_val_acc[i] = val_res['acc']
            if train_res['acc'] > best_train_acc[i]:
                best_train_acc[i] = train_res['acc']
                torch.save(models[i].state_dict(), os.path.join('result', model_names[i] + f'_{i}_best.pth'))

            # save training information in wandb
            wandb.log({
                "{}_{}_{}_{}_ceLoss".format(i, model_names[i], args.dataset, 'train'): train_res['ce_loss'],
                "{}_{}_{}_{}_klLoss".format(i, model_names[i], args.dataset, 'train'): train_res['kl_loss'],
                "{}_{}_{}_{}_l2Loss".format(i, model_names[i], args.dataset, 'train'): train_res['l2_loss'],
                "{}_{}_{}_{}_loss".format(i, model_names[i], args.dataset, 'train'): train_res['loss'],
                "{}_{}_{}_{}_acc".format(i, model_names[i], args.dataset, 'train'): train_res['acc'],
                "{}_{}_{}_{}_loss".format(i, model_names[i], args.dataset, 'val'): val_res['loss'],
                "{}_{}_{}_{}_acc".format(i, model_names[i], args.dataset, 'val'): val_res['acc'],
            }, step=epoch)

        with open(os.path.join(SAVE_PATH, 'competitive_distillation_result.csv'), 'a', encoding='utf-8', newline='') as f:
            fileName = ['model_name', 'train_acc', 'val_acc']
            writer = csv.DictWriter(f, fieldnames=fileName)
            if epoch == 0:
                writer.writeheader()
            for i in range(len(model_names)):
                writer.writerow({'model_name':model_names[i],
                                'train_acc': train_res['acc'],
                                'val_acc': val_res['acc'],})

    wandb.finish()

    # record best performance
    with open(os.path.join(SAVE_PATH, 'best_result.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(models)):
            f.write(f'model_name: {i}_{model_names[i]}  train_acc: {best_train_acc[i]:.4f}  val_acc: {best_val_acc[i]:.4f}\n')
    print(f'{model_names} best_train_acc:{best_train_acc} best_val_acc:{best_val_acc}')

def ComDis_(model_str, proj_name, SAVE_PATH, is_augmentation):
    args = parse_args()
    train(args, model_str, proj_name, SAVE_PATH, is_augmentation=is_augmentation)





