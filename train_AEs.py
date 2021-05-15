import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.nn.functional as F
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torchvision import models

from dataset import CIFAR
from models import AE, AE_ImageNet, Alexnet_FE, VAE
from utils import accuracy, AE_loss, get_task_dir, load_model, make_dir, save_model

torch.random.manual_seed(13)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ae_type = "AE"  # AE, VAE, AE_ImageNet
models_dir = f"models/{ae_type}s"
make_dir(models_dir)

classes_split = [range(20 * i, (i + 1) * 20) for i in range(5)]
batch_size = 128


def train(task_dir, classes, hidden_dim, num_epochs, lr):
    train_data = CIFAR(
        "CIFAR/train",
        classes=classes,
        image_size=32 if ae_type is not "AE_ImageNet" else 224,
    )
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = CIFAR(
        "CIFAR/test",
        classes=classes,
        image_size=32 if ae_type is not "AE_ImageNet" else 224,
    )
    dataloader_test = DataLoader(test_data, batch_size=batch_size)

    if ae_type == "AE":
        model = AE(32 * 32 * 3, hidden_dim).to(device)
        criterion = nn.MSELoss(reduction="sum")
    if ae_type == "VAE":
        MSELoss = nn.MSELoss(reduction="sum")
        model = VAE(32 * 32 * 3, hidden_dim).to(device)
        criterion = lambda output, imgs: MSELoss(output[0], imgs) - 0.5 * torch.sum(
            1 + output[2] - output[1].pow(2) - output[2].exp()
        )
    if ae_type == "AE_ImageNet":
        model = AE_ImageNet(hidden_dim).to(device)
        criterion = nn.MSELoss(reduction="sum")
        pretrained_alexnet = models.alexnet(pretrained=True)
        feature_extractor = Alexnet_FE(pretrained_alexnet).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_all = 0.0, 0.0
        for imgs, _ in dataloader_train:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            if ae_type == "AE_ImageNet":
                imgs = feature_extractor(imgs)
                imgs = imgs.view(imgs.shape[0], -1)
                imgs = torch.sigmoid(imgs)
            output = model(imgs)
            loss = criterion(output, imgs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.shape[0]
            train_all += imgs.shape[0]

        train_loss = train_loss / train_all
        if ae_type == "AE_ImageNet":
            test_loss = AE_loss(model, dataloader_test, criterion, feature_extractor)
        else:
            test_loss = AE_loss(model, dataloader_test, criterion)
        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}"
        )

    return model, optimizer


def is_img_matched_to_task(task, all_tasks, classes, hidden_dim):
    test_data = CIFAR(
        f"CIFAR/test",
        classes=classes,
        image_size=32 if ae_type is not "AE_ImageNet" else 224,
    )
    dataloader_test = DataLoader(test_data, batch_size=batch_size)

    if ae_type == "AE":
        model = AE(32 * 32 * 3, hidden_dim).to(device)
        criterion = nn.MSELoss(reduction="sum")
    if ae_type == "VAE":
        model = VAE(32 * 32 * 3, hidden_dim).to(device)
        MSELoss = nn.MSELoss(reduction="sum")
        criterion = lambda output, imgs: MSELoss(output[0], imgs) - 0.5 * torch.sum(
            1 + output[2] - output[1].pow(2) - output[2].exp()
        )
    if ae_type == "AE_ImageNet":
        model = AE_ImageNet(hidden_dim).to(device)
        criterion = nn.MSELoss(reduction="sum")
        pretrained_alexnet = models.alexnet(pretrained=True)
        feature_extractor = Alexnet_FE(pretrained_alexnet).to(device)

    is_matched = list()
    pred, true = list(), list()
    with torch.no_grad():

        for imgs, labels in dataloader_test:
            imgs = imgs.to(device)
            loss = torch.empty(all_tasks)
            for i in range(all_tasks):
                model_path = f"{get_task_dir(models_dir, i)}/{ae_type}.pch"
                model = load_model(model_path, model)
                model.eval()
                if ae_type == "AE_ImageNet":
                    imgs_i = feature_extractor(imgs.clone())
                    imgs_i = imgs_i.view(imgs_i.shape[0], -1)
                    imgs_i = torch.sigmoid(imgs_i)
                    output = model(imgs_i)
                    loss[i] = criterion(output, imgs_i)
                else:
                    output = model(imgs)
                    loss[i] = criterion(output, imgs)

            is_matched += [loss.argmin() == task]

    return torch.tensor(is_matched)


def matching_accuracy(classes_split, max_task, hidden_dim):
    matching_acc = []
    preds, trues = [], []
    for task, classes in enumerate(classes_split[0:max_task]):
        task_dir = get_task_dir(models_dir, task)
        is_matched = is_img_matched_to_task(task, max_task, classes, hidden_dim)
        acc = is_matched.float().mean()
        matching_acc += [acc]
    print(f"Matched tasks {np.mean(matching_acc)} | {[i.item() for i in matching_acc]}")


lr = 0.001
hidden_dim = 64
num_epochs = 120

for task, classes in enumerate(classes_split):
    print(f"Task: {task}")
    task_dir = get_task_dir(models_dir, task)
    make_dir(task_dir)

    model, optimizer = train(task_dir, classes, hidden_dim, num_epochs, lr=lr)
    save_model(f"{task_dir}/{ae_type}.pch", model, optimizer)

    matching_accuracy(classes_split, task + 1, hidden_dim)
