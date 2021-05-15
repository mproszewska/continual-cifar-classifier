import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CIFAR
from models import AlexNet, AE, Generator, VAE
from utils import (
    accuracy,
    get_model_dir,
    get_task_dir,
    load_model,
    make_dir,
    save_model,
    tasks_relatedness,
)

torch.random.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dir = "models/main"
make_dir(models_dir)

ae_type = "AE"  # AE or VAE
ae_dir = f"models/{ae_type}s"

hidden_dim = 64

classes = range(100)
batch_size = 128
relatedness_threshold = 0.95
retrain = False
GR = None  # real, noise, VAE or GAN
gr_dir = f"models/VAEs"
z_dim = 128

def get_model_task_mapping(all_tasks):
    model_task_mapping = torch.zeros(all_tasks, dtype=int)
    with open(f"{models_dir}/model_task_mapping.txt", "r") as f:
        for i, line in enumerate(f):
            tasks = [int(t) for t in line.split(",")]
            model_task_mapping[tasks] = i
    return model_task_mapping


def update_model_task_mapping(task, model_id):
    model_task_mapping = torch.zeros(task + 1, dtype=int)
    if task > 0:
        model_task_mapping[:-1] = get_model_task_mapping(task)
    model_task_mapping[task] = model_id
    with open(f"{models_dir}/model_task_mapping.txt", "w") as f:
        for model_id in range(model_task_mapping.max() + 1):
            tasks = [
                str(t) for t in range(task + 1) if model_task_mapping[t] == model_id
            ]
            f.write(",".join(tasks) + "\n")


def is_img_matched_to_task(task, all_tasks, classes, hidden_dim):
    model_task_mapping = get_model_task_mapping(all_tasks)

    test_data = CIFAR(f"CIFAR/test", classes=classes, image_size=32)
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

    is_matched = list()
    with torch.no_grad():
        for imgs, _ in dataloader_test:
            imgs = imgs.to(device)
            loss = torch.empty(all_tasks)
            for i in range(all_tasks):
                model_path = f"{get_task_dir(ae_dir, i)}/{ae_type}.pch"
                model = load_model(model_path, model)
                output = model(imgs)
                loss[i] = criterion(output, imgs)

            is_matched += [
                model_task_mapping[loss.argmin()] == model_task_mapping[task]
            ]

    return torch.tensor(is_matched)


def get_related_model(task, classes, hidden_dim):
    """
    Returns number of model that is most related to current task and a metric that measures that.
	"""
    train_data = CIFAR(f"CIFAR/train", classes=classes, image_size=32)
    dataloader_train = DataLoader(train_data, batch_size=batch_size)

    if ae_type == "AE":
        model = AE(32 * 32 * 3, hidden_dim).to(device)
        criterion = nn.MSELoss(reduction="sum")
    if ae_type == "VAE":
        MSELoss = nn.MSELoss(reduction="sum")
        model = VAE(32 * 32 * 3, hidden_dim).to(device)
        criterion = lambda output, imgs: MSELoss(output[0], imgs) - 0.5 * torch.sum(
            1 + output[2] - output[1].pow(2) - output[2].exp()
        )

    relatedness = [0] * task
    model_task_mapping = get_model_task_mapping(task)
    for i in range(task, -1, -1):
        model_path = f"{get_task_dir(ae_dir, i)}/{ae_type}.pch"
        model = load_model(model_path, model)
        train_loss, train_all = 0.0, 0.0

        model.eval()
        with torch.no_grad():
            for imgs, _ in dataloader_train:
                imgs = imgs.to(device)
                output = model(imgs)
                loss = criterion(output, imgs)
                train_loss += loss.item() * imgs.shape[0]
                train_all += imgs.shape[0]

        mean_loss = train_loss / train_all
        if i == task:
            rec_error_task = mean_loss
        else:
            relatedness[i] = tasks_relatedness(rec_error_task, mean_loss)

    relatedness_per_model = [0] * (max(model_task_mapping) + 1)
    for i in range(len(relatedness_per_model)):
        relatedness_per_model[i] = np.mean(
            [relatedness[t] for t in range(task) if model_task_mapping[t] == i]
        )

    most_related_model = np.argmax(relatedness_per_model)
    best_relatedness = relatedness_per_model[most_related_model]
    print(
        f"Most related model {most_related_model} with relatedness {best_relatedness:.4f}"
    )
    related_classes = []
    for t in range(task):
        if model_task_mapping[t] == most_related_model:
            related_classes += classes_split[t]
    return most_related_model, best_relatedness, related_classes


def copy_related_AlexNet(model, related_model):
    params = model.named_parameters()
    related_params = related_model.named_parameters()

    dict_params = dict(params)

    for name, param in related_params:
        if name in dict_params:
            if not name.startswith("net.22"):
                dict_params[name].data.copy_(param.data)
            else:
                dict_params[name].data[: param.data.shape[0]].copy_(param.data)


def soft_target_loss(preds, labels, MSE=True, T=2):
    if MSE:
        criterion = nn.MSELoss(reduction="sum")
        return criterion(preds, labels)
    else:
        preds = F.softmax(preds, dim=1).pow(1 / T)
        labels = F.softmax(labels, dim=1).pow(1 / T)

        sum_preds = torch.sum(preds, dim=1)
        sum_labels = torch.sum(preds, dim=1)

        sum_preds_ref = torch.transpose(sum_preds.repeat(preds.size(1), 1), 0, 1)
        sum_labels_ref = torch.transpose(sum_labels.repeat(labels.size(1), 1), 0, 1)

        preds = preds / sum_preds_ref
        labels = labels / sum_labels_ref

        loss = torch.sum(-1 * preds * torch.log(labels), dim=1)

        return torch.sum(loss, dim=0)


def GR_GAN(task, related_model, model, related_classes, sigma=1.):
    related_output = torch.empty(batch_size, len(related_classes), device=device)
    output_related_classes = torch.empty(
        batch_size, len(related_classes), device=device
    )
    for t in [c // 20 for c in related_classes][::20]:
        gan = Generator(z_dim, (3, 32, 32)).to(device)
        model_path = f"{get_task_dir(gr_dir, t)}/Generator.pch"
        gan = load_model(model_path, gan)
        noise = torch.normal(
            torch.zeros(batch_size, z_dim, 1, 1),
            sigma * torch.ones(batch_size, z_dim, 1, 1),
        ).to(device)
        with torch.no_grad():
            generated_imgs = gan(noise)
            generated_imgs = transforms.Resize(224)(generated_imgs)
            related_output[:, t * 20 : (t + 1) * 20] = related_model(generated_imgs)[
                :, t * 20 : (t + 1) * 20
            ]
            output_related_classes[:, t * 20 : (t + 1) * 20] = model(generated_imgs)[
                :, t * 20 : (t + 1) * 20
            ]
            del gan
    return related_output, output_related_classes


def GR_VAE(task, related_model, model, related_classes, sigma=1.):
    related_output = torch.empty(batch_size, len(related_classes), device=device)
    output_related_classes = torch.empty(
        batch_size, len(related_classes), device=device
    )
    for t in [c // 20 for c in related_classes][::20]:
        vae = VAE(32 * 32 * 3, hidden_dim).to(device)
        model_path = f"{get_task_dir(gr_dir, t)}/VAE.pch"
        vae = load_model(model_path, vae)
        noise = torch.normal(
            torch.zeros(batch_size, hidden_dim),
            sigma * torch.ones(batch_size, hidden_dim),
        ).to(device)
        with torch.no_grad():
            noise = vae.project(noise).view(noise.shape[0], -1, 4, 4)
            generated_imgs = vae.decoder(noise).reshape(batch_size, 3, 32, 32)
            generated_imgs = transforms.Resize(224)(generated_imgs)
            related_output[:, t * 20 : (t + 1) * 20] = related_model(generated_imgs)[
                :, t * 20 : (t + 1) * 20
            ]
            output_related_classes[:, t * 20 : (t + 1) * 20] = model(generated_imgs)[
                :, t * 20 : (t + 1) * 20
            ]
            del vae
    return related_output, output_related_classes


def GR_noise(task, related_model, model, related_classes, sigma=1.):
    generated_imgs = torch.normal(
        torch.zeros(batch_size, 3, 224, 224),
        sigma * torch.ones(batch_size, 3, 224, 224),
    ).to(device)
    with torch.no_grad():
        related_output = related_model(generated_imgs)
    output_related_classes = model(generated_imgs)[:, : len(related_classes)]
    return related_output, output_related_classes


def GR_real(task, related_model, model, related_classes, dataloader_GR):
    related_output = torch.empty(batch_size, len(related_classes), device=device)
    output_related_classes = torch.empty(
        batch_size, len(related_classes), device=device
    )
    for i, t in enumerate([c // 20 for c in related_classes][::20]):
        with torch.no_grad():
            generated_imgs = next(iter(dataloader_GR[t]))[0].to(device)
            related_output[:, t * 20 : (t + 1) * 20] = related_model(generated_imgs)[
                :, t * 20 : (t + 1) * 20
            ]
            output_related_classes[:, t * 20 : (t + 1) * 20] = model(generated_imgs)[
                :, t * 20 : (t + 1) * 20
            ]

    return related_output, output_related_classes


def train_model(current_model_id, task, classes, num_epochs, lr, LwF_alpha):
    train_data = CIFAR("CIFAR/train", classes=classes)
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if task > 0:
        related_model_id, relatedness, related_classes = get_related_model(
            task, classes, hidden_dim
        )

    if task > 0 and relatedness > relatedness_threshold:
        print("Adjusting model for related task")
        related_model_path = f"{get_model_dir(models_dir, related_model_id)}/Model.pch"
        test_data = CIFAR("CIFAR/test", classes=related_classes + classes)
        dataloader_test = DataLoader(test_data, batch_size=batch_size)

        related_model = AlexNet(num_classes=len(related_classes)).to(device)
        related_model = load_model(related_model_path, related_model)

        if GR == "real":
            GR_data = [None] * task
            dataloader_GR = [None] * task
            for i, t in enumerate([c // 20 for c in related_classes][::20]):
                GR_data[i] = CIFAR(
                    "CIFAR/train", classes=[*range(t * 20, (t + 1) * 20)]
                )
                dataloader_GR[i] = DataLoader(
                    GR_data[i], batch_size=batch_size, shuffle=True
                )

        model = AlexNet(num_classes=len(related_classes + classes)).to(device)

        if not retrain:
            copy_related_AlexNet(model, related_model)
        del related_model

        related_model = AlexNet(num_classes=len(related_classes)).to(device)
        related_model = load_model(related_model_path, related_model)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.004)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)

        criterion = nn.CrossEntropyLoss(reduction="sum")
        softmax_intervals = [20] * (len(related_classes + classes) // 20)
        for epoch in range(1, num_epochs + 1):
            model.train()
            related_model.eval()
            train_loss, train_acc, train_all = 0.0, 0.0, 0.0
            for imgs, labels in dataloader_train:
                imgs, labels = imgs.to(device), labels.to(device)
                labels_c = labels.clone()
                for i, c in enumerate(classes):
                    labels[labels_c == c] = i

                if GR is None:
                    with torch.no_grad():
                        related_output = related_model(imgs)
                if GR == "GAN":
                    related_output, output_related_classes = GR_GAN(
                        task, related_model, model, related_classes
                    )
                if GR == "VAE":
                    related_output, output_related_classes = GR_VAE(
                        task, related_model, model, related_classes
                    )
                if GR == "noise":
                    related_output, output_related_classes = GR_noise(
                        task, related_model, model, related_classes
                    )
                if GR == "real":
                    related_output, output_related_classes = GR_real(
                        task, related_model, model, related_classes, dataloader_GR
                    )

                model.train()
                optimizer.zero_grad()
                output = model(imgs)
                if GR is None:
                    output_related_classes = output[:, : len(related_classes)]
                output_classes = output[:, len(related_classes) :]
                related_loss = soft_target_loss(output_related_classes, related_output)
                new_loss = criterion(output_classes, labels)
                loss = LwF_alpha * related_loss + new_loss
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                train_all += imgs.shape[0]
            lr_scheduler.step()
            if epoch % 1 == 0:

                print(
                    f"Epoch: {epoch} | Train loss: {train_loss/train_all:.4f} | Test accuracy: {accuracy(model, dataloader_test, related_classes+classes, softmax_intervals=softmax_intervals)}"
                )

        return related_model_id, model, optimizer
    else:
        test_data = CIFAR("CIFAR/test", classes=classes)
        dataloader_test = DataLoader(test_data, batch_size=batch_size)

        model = AlexNet(num_classes=len(classes)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.004)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        criterion = nn.CrossEntropyLoss(reduction="sum")
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss, train_acc, train_all = 0.0, 0.0, 0.0
            for imgs, labels in dataloader_train:
                imgs, labels = imgs.to(device), labels.to(device)
                labels_c = labels.clone()
                for i, c in enumerate(classes):
                    labels[labels_c == c] = i

                optimizer.zero_grad()
                output = model(imgs)
                loss = criterion(output, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                train_acc += (output.argmax(dim=-1) == labels).sum().item()
                train_all += imgs.shape[0]
            lr_scheduler.step()
            print(
                f"Epoch: {epoch} | Train loss: {train_loss/train_all:.4f} | Train accuracy: {train_acc/train_all:.4f} | Test accuracy: {accuracy(model, dataloader_test, classes)}"
            )

        return current_model_id, model, optimizer


def full_accuracy(classes_split, max_task, current_model_id, hidden_dim):
    accs = []
    for task, classes in enumerate(classes_split[0:max_task]):
        model_task_mapping = get_model_task_mapping(max_task)
        model_id = model_task_mapping[task]
        is_matched = is_img_matched_to_task(task, max_task, classes, hidden_dim)
        all_model_classes = []
        for t in range(max_task):
            if model_id == model_task_mapping[t]:
                all_model_classes += classes_split[t]
        print(f"Matched {is_matched.sum()/is_matched.shape[0]}")

        model_path = f"{get_model_dir(models_dir, model_id)}/Model.pch"
        model = AlexNet(num_classes=len(all_model_classes)).to(device)
        model = load_model(model_path, model)

        test_data = CIFAR("CIFAR/test", classes=classes)
        dataloader_test = DataLoader(test_data, batch_size=batch_size)
        acc = accuracy(
            model,
            dataloader_test,
            all_model_classes,
            is_matched,
            [20] * (len(all_model_classes) // 20),
        )
        accs += [acc]
        print(f"Test accuracy: {acc}")
    print(f"Overall accuracy: {np.mean(accs)}")


classes_split = [[*range(20 * i, (i + 1) * 20)] for i in range(5)]
model_task_mapping_file = f"{models_dir}/model_task_mapping.txt"
if not os.path.exists(model_task_mapping_file):
    os.mknod(model_task_mapping_file)

current_model_id = 0
for task, classes in enumerate(classes_split):
    print(f"Task Number {task}")
    new_model_id, model, optimizer = train_model(
        current_model_id, task, classes, num_epochs=90, lr=5e-3, LwF_alpha=0.01
    )
    model_dir = get_model_dir(models_dir, new_model_id)
    make_dir(model_dir)
    save_model(f"{model_dir}/Model.pch", model, optimizer)
    update_model_task_mapping(task, new_model_id)

    if new_model_id == current_model_id:
        current_model_id += 1

    full_accuracy(classes_split, task + 1, task, hidden_dim)
