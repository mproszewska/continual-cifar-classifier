import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import CIFAR
from models import AlexNet
from utils import accuracy, make_dir, save_model

torch.random.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = range(100)
batch_size = 128

train_data = CIFAR("CIFAR/train", classes=classes)
dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = CIFAR("CIFAR/test", classes=classes)
dataloader_test = DataLoader(test_data, batch_size=batch_size)

model_dir = "models/joint_training"
make_dir(model_dir)


def train(num_epochs, lr, step_size):
    model = AlexNet(num_classes=len(classes), p=0.5).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=4e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_acc, train_all = 0.0, 0.0, 0.0
        for imgs, labels in dataloader_train:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (output.argmax(dim=-1) == labels).sum().item()
            train_all += imgs.shape[0]
        lr_scheduler.step()

        train_loss, train_acc = train_loss / train_all, train_acc / train_all
        test_acc, _ = accuracy(model, dataloader_test, classes)
        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}"
        )
    return model, optimizer


model, optimizer = train(num_epochs=90, lr=5e-4, step_size=30)
save_model(f"{model_dir}/AlexNet.pch", model, optimizer)
