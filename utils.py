import os
import torch


def accuracy(model, dataloader_test, classes, classes_before=[], is_matched=None):
    model.eval()
    device = next(model.parameters()).device
    test_all, test_acc, test_acc_if_matched = 0, 0.0, 0
    with torch.no_grad():
        for imgs, labels in dataloader_test:
            imgs, labels = imgs.to(device), labels.to(device)
            for i, c in enumerate(classes):
                labels[labels == c] = i
            output = model(imgs)
            output_classes = output[
                :, len(classes_before) : len(classes_before) + len(classes)
            ]
            is_predicted = output_classes.argmax(dim=-1) == labels
            test_acc_if_matched += (is_predicted).sum().item()
            if is_matched is not None:
                mask = is_matched[test_all : test_all + imgs.shape[0]]
                is_predicted[~mask] = 0
            test_acc += (is_predicted).sum().item()
            test_all += imgs.shape[0]
    return test_acc / test_all, test_acc_if_matched / test_all


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


def expert_gate_loss(model, dataloader_test, criterion):
    test_loss, test_all = 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for imgs, _ in dataloader_test:
            imgs = imgs.to(device).flatten(start_dim=1)
            output = model(imgs)
            loss = criterion(output, imgs)
            test_loss += loss.item() * imgs.shape[0]
            test_all += imgs.shape[0]
    return test_loss / test_all
