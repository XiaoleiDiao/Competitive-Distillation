import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
x1 = torch.randn(1,2,3,3).cuda()
import time
import train_competitive_distillation
import train_mutual_learning
import utils

def train_(model_str, proj_name, is_augmentation=True):
    SAVE_PATH = utils.get_save_path()
    print('SAVE_PATH: ', SAVE_PATH)
    train_competitive_distillation.ComDis_(model_str, 'competitive_distillation_' + proj_name, SAVE_PATH, is_augmentation=is_augmentation)
    train_mutual_learning.mutual_(model_str, 'mutual_' + proj_name, SAVE_PATH, is_augmentation=is_augmentation)


if __name__ == '__main__':
    model_str = [
                    'resnet32, resnet32',
                    # 'ceit, ViT',

                ]

    proj_name = [
                    'multy_raodong_res32_res32_every_10_batch',
                    # 'multy_raodong_ceit_viT_every_10_batch',
                ]

    for i in range(0, 1):
        print('*' * 100)
        train_(model_str[i], proj_name[i])
