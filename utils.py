import numpy as np
import os
import torch
import torch.nn.functional as F

from torch.autograd import Variable


def accuracy(
    model,
    dataloader_test,
    classes,
    is_matched=None,
    softmax_intervals=None,
    task_interval=None,
):
    model.eval()
    device = next(model.parameters()).device
    test_all, test_acc, test_acc_if_matched = 0, 0.0, 0
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(dataloader_test):
            imgs, labels = imgs.to(device), labels.to(device)
            labels_c = labels.clone()
            for i, c in enumerate(classes):
                labels[labels_c == c] = i
            output = model(imgs)
            if softmax_intervals is not None:
                start = 0
                for i, si in enumerate(softmax_intervals):
                    output[:, start : start + si] = F.softmax(
                        output[:, start : start + si], dim=1
                    )
                    if task_interval is not None and task_interval == i:
                        output = output[:, start : start + si]
                        break
                    start += si
            is_predicted = output.argmax(dim=-1) == labels
            if is_matched is not None:
                mask = is_matched[idx] * torch.ones(imgs.shape[0]).bool()
                is_predicted[~mask] = 0
            test_acc += (is_predicted).sum().item()
            test_all += imgs.shape[0]
    return test_acc / test_all


def save_model(path, model, optimizer):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def load_model(path, model):
    loaded_state = torch.load(path)
    model.load_state_dict(loaded_state["model"])
    return model


def load_optimizer(path, optimizer):
    loaded_state = torch.load(path)
    optimizer.load_state_dict(loaded_state["optimizer"])
    return optimizer


def rename_labels(labels, classes):
    labels_copy = labels.clone()
    for i, c in enumerate(classes):
        labels[labels_copy == c] = i
    return labels


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def tasks_relatedness(rec_error_first, rec_error_second):
    return 1 - ((rec_error_second - rec_error_first) / rec_error_first)


def get_task_dir(model_dir, task):
    return f"{model_dir}/task_{task:02}"


def get_model_dir(model_dir, model):
    return f"{model_dir}/model_{model:02}"


def AE_loss(model, dataloader_test, criterion, feature_extractor=None):
    test_loss, test_all = 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for imgs, _ in dataloader_test:
            imgs = imgs.to(device)
            if feature_extractor is not None:
                imgs = feature_extractor(imgs)
                imgs = imgs.view(imgs.shape[0], -1)
                imgs = torch.sigmoid(imgs)
            output = model(imgs)
            loss = criterion(output, imgs)
            test_loss += loss.item() * imgs.shape[0]
            test_all += imgs.shape[0]
    return test_loss / test_all


def GAN_loss(model_G, model_D, dataloader_test, criterion, z_dim):
    test_G_loss, test_D_loss, test_all = 0, 0, 0
    device = next(model_G.parameters()).device
    real_label, fake_label = 1.0, 0.0

    with torch.no_grad():
        for imgs, _ in dataloader_test:
            imgs = imgs.to(device)
            valid = Variable(
                torch.Tensor(imgs.shape[0], 1).fill_(real_label), requires_grad=False
            ).to(device)
            fake = Variable(
                torch.Tensor(imgs.shape[0], 1).fill_(fake_label), requires_grad=False
            ).to(device)
            noise = Variable(
                torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], z_dim, 1, 1)))
            ).to(device)

            real_loss = criterion(model_D(imgs), valid)
            gen_imgs = model_G(noise)
            fake_loss = criterion(model_D(gen_imgs.detach()), fake)
            loss_D = real_loss + fake_loss

            gen_imgs = model_G(noise)
            output = model_D(gen_imgs)
            D_x = output.mean().item()
            loss_G = criterion(output, valid)

            test_D_loss += real_loss.item() * imgs.shape[0]
            test_G_loss += loss_D.item() * imgs.shape[0]
            test_all += imgs.shape[0]
    return test_G_loss / test_all, test_D_loss / test_all
