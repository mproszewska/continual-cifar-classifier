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

from dataset import CIFAR
from models import AE
from utils import (
    expert_gate_loss,
    get_task_dir,
    load_model,
    make_dir,
    save_model,
    tasks_relatedness,
)

torch.random.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gate_type = "AE"  # AE or VAE
models_dir = f"models/{gate_type}_gates"
make_dir(models_dir)

classes_split = [range(10 * i, (i + 1) * 10) for i in range(10)]
batch_size = 128


def train(task_dir, classes, hidden_dim, num_epochs, lr):
    train_data = CIFAR("CIFAR/train", classes=classes, image_size=32)
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = CIFAR("CIFAR/test", classes=classes, image_size=32)
    dataloader_test = DataLoader(test_data, batch_size=batch_size)

    model = AE(32 * 32 * 3, hidden_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    criterion = nn.MSELoss()
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_all = 0.0, 0.0
        for imgs, _ in dataloader_train:
            imgs = imgs.to(device).flatten(start_dim=1)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, imgs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.shape[0]
            train_all += imgs.shape[0]

        train_loss = train_loss / train_all
        test_loss = expert_gate_loss(model, dataloader_test, criterion)
        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}"
        )
        
    return model, optimizer


def is_img_matched_to_task(task, all_tasks, classes, hidden_dim):
    test_data = CIFAR(f"CIFAR/test", classes=classes, image_size=32)
    dataloader_test = DataLoader(test_data, batch_size=batch_size)

    criterion = nn.MSELoss(reduction="none")
    is_matched = list()
    pred = []
    with torch.no_grad():
        for imgs, _ in dataloader_test:
            imgs = imgs.to(device).flatten(start_dim=1)

            loss = torch.empty(imgs.shape[0], all_tasks)
            for i in range(all_tasks):
                model_path = f"{get_task_dir(models_dir, i)}/AE.pch"
                model = AE(32 * 32 * 3, hidden_dim).to(device)
                model = load_model(model_path, model)
                output = model(imgs)
                loss[:, i] = (
                    criterion(output, imgs).reshape(imgs.shape[0], -1).mean(dim=1)
                )

            is_matched += [loss.mean(dim=0).argmin() == task]
    return torch.tensor(is_matched)


def matching_accuracy(classes_split, max_task, hidden_dim):
    matching_acc = []
    for task, classes in enumerate(classes_split[0:max_task]):
        task_dir = get_task_dir(models_dir, task)
        is_matched = is_img_matched_to_task(task, max_task, classes, hidden_dim)
        acc = is_matched.sum() / is_matched.shape[0]
        matching_acc += [acc]
    print(f"Matched {np.mean(matching_acc)}")


lr = 0.001
hidden_dim = 64
num_epochs = 40
for task, classes in enumerate(classes_split):
    print(f"Task: {task}")

    task_dir = get_task_dir(models_dir, task)
    make_dir(task_dir)

    model, optimizer = train(task_dir, classes, hidden_dim, num_epochs, lr=lr)
    save_model(f"{task_dir}/AE.pch", model, optimizer)

    matching_accuracy(classes_split, task + 1, hidden_dim)
