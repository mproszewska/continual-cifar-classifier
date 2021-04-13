import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CIFAR
from models import AlexNet
from utils import accuracy, get_task_dir, make_dir, save_model, rename_labels

torch.random.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dir = "models/task_oracle"
make_dir(models_dir)

classes_split = [range(10 * i, (i + 1) * 10) for i in range(10)]
batch_size = 128


def train(classes, num_epochs, lr, step_size, dropout_p):
    train_data = CIFAR("CIFAR/train", classes=classes)
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = CIFAR("CIFAR/test", classes=classes)
    dataloader_test = DataLoader(test_data, batch_size=batch_size)

    model = AlexNet(num_classes=len(classes), p=dropout_p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=4e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_all, train_acc = 0.0, 0.0, 0
        for imgs, labels in dataloader_train:
            imgs, labels = imgs.to(device), labels.to(device)
            labels = rename_labels(labels, classes)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_all += imgs.shape[0]
            train_acc += (output.argmax(dim=-1) == labels).sum().item()

        lr_scheduler.step()
        train_loss = train_loss / train_all
        train_acc = train_acc / train_all
        test_acc, _ = accuracy(model, dataloader_test, classes)
        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}"
        )
    return model, optimizer


for task, classes in enumerate(classes_split):
    print(f"Task: {task}")
    model, optimizer = train(
        classes, num_epochs=90, lr=5e-4, step_size=30, dropout_p=0.7
    )
    task_dir = get_task_dir(models_dir, task)
    make_dir(task_dir)
    save_model(f"{task_dir}/AlexNet.pch", model, optimizer)
