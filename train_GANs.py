import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.nn.functional as F
import torchvision.utils as vutils

from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import CIFAR
from models import Discriminator, Generator
from utils import GAN_loss, get_task_dir, load_model, make_dir, save_model

torch.random.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dir = f"models/GANs"
make_dir(models_dir)

classes_split = [range(20 * i, (i + 1) * 20) for i in range(5)]


def train(task_dir, classes, z_dim, num_epochs, lr):
    train_data = CIFAR("CIFAR/train", classes=classes, image_size=32)
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = CIFAR("CIFAR/test", classes=classes, image_size=32)
    dataloader_test = DataLoader(test_data, batch_size=batch_size)

    model_G = Generator(z_dim, (3, 32, 32)).to(device)
    model_D = Discriminator((3, 32, 32)).to(device)
    real_label, fake_label = float(1), float(0)
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()

    for epoch in range(1, num_epochs + 1):
        model_G.train()
        model_D.train()
        train_G_loss, train_D_loss, train_all, test_loss, test_all = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        for imgs, _ in dataloader_train:
            imgs = imgs.to(device)
            valid = Variable(
                torch.Tensor(imgs.size(0), 1).fill_(real_label), requires_grad=False
            ).to(device)
            fake = Variable(
                torch.Tensor(imgs.size(0), 1).fill_(fake_label), requires_grad=False
            ).to(device)

            optimizer_D.zero_grad()

            real_loss = criterion(model_D(imgs), valid)
            noise = Variable(
                torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], z_dim, 1, 1)))
            ).to(device)
            gen_imgs = model_G(noise)
            fake_loss = criterion(model_D(gen_imgs.detach()), fake)

            loss_D = real_loss + fake_loss
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(model_D.parameters(), 1.0)
            optimizer_D.step()

            optimizer_G.zero_grad()
            gen_imgs = model_G(noise)
            output = model_D(gen_imgs)
            loss_G = criterion(output, valid)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(model_G.parameters(), 1.0)
            optimizer_G.step()

            train_D_loss += real_loss.item() * imgs.shape[0]
            train_G_loss += loss_G.item() * imgs.shape[0]
            train_all += imgs.shape[0]

        train_G_loss = train_G_loss / train_all
        train_D_loss = train_D_loss / train_all
        test_G_loss, test_D_loss = GAN_loss(
            model_G, model_D, dataloader_test, criterion, z_dim
        )
        print(
            f"Epoch: {epoch} | Train loss : G {train_G_loss:.4f}  D {train_D_loss:.4f} | Test loss: G {test_G_loss:.4f} D {test_D_loss:.4f}"
        )
    return model_G, model_D, optimizer_G, optimizer_D


lr = 0.001
z_dim = 128
num_epochs = 150
batch_size = 128

for task, classes in enumerate(classes_split):
    print(f"Task: {task}")

    task_dir = get_task_dir(models_dir, task)
    make_dir(task_dir)

    model_G, model_D, optimizer_G, optimizer_D = train(
        task_dir, classes, z_dim, num_epochs, lr=lr
    )
    save_model(f"{task_dir}/Generator.pch", model_G, optimizer_G)
    save_model(f"{task_dir}/Discriminator.pch", model_D, optimizer_D)
