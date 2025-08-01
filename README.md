# Competitive Distillation for Image Classification

📢 **This work has been accepted at ICCV 2025!**  
📄 Paper: [https://arxiv.org/pdf/2506.23285](https://arxiv.org/pdf/2506.23285)

This repository contains code for training and evaluating deep neural networks using **competitive distillation** and other collaborative training strategies. It supports various model architectures including ResNet, MobileNet, Vision Transformers (ViT), and CeiT.

## 🔍 Overview

Knowledge distillation has become a popular technique to transfer knowledge from a large teacher model to a smaller student model. This project explores **competitive and mutual distillation**, where multiple models learn collaboratively by sharing knowledge during training.

## ✨ Features

- Supports multiple training paradigms:
  - Vanilla training (`train.py`)
  - Mutual learning (`train_mutual_learning.py`)
  - Competitive distillation (`train_competitive_distillation.py`)
  - CeiT-specific training (`train_ceit.py`)
- Includes various model architectures: ResNet, WideResNet, MobileNet, Inception, ViT, CeiT
- Customizable dataset loader and utility functions

## 📁 Dataset Preparation
The datasets/dataset.py script supports standard datasets. Modify or extend it to load your own dataset if needed.

## 🚀 Training
1. Standard Training
python train.py --model resnet --dataset cifar10

2. Mutual Learning
python train_mutual_learning.py --model1 resnet --model2 mobilenet --dataset cifar10

3. Competitive Distillation
python train_competitive_distillation.py --teacher resnet --student vit --dataset cifar10


## 🧱 Project Structure
  .
  ├── train.py                         # Vanilla training
  ├── train_mutual_learning.py        # Mutual learning training
  ├── train_competitive_distillation.py # Competitive distillation
  ├── train_ceit.py                   # CeiT-specific training
  ├── demo.py                         # Example usage or visualization
  ├── datasets/
  │   └── dataset.py                  # Dataset loading logic
  ├── models/
  │   ├── resnet.py, vit.py, CeiT.py  # Model definitions
  ├── utils.py                        # Utility functions


