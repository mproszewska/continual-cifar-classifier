import numpy as np
import pickle
import torch
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset


class CIFAR(Dataset):
    def __init__(self, filename, classes=range(100), image_size=224):
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]
                ),
            ]
        )
        with open(filename, "rb") as f:
            content = pickle.load(f, encoding="bytes")
            data = content[b"data"]
            labels = content[b"fine_labels"]

        self.data, self.labels = list(), list()

        for img, label in zip(data, labels):
            if label in classes:
                img = np.moveaxis(img.reshape(3, 32, 32), [0, 1, 2], [2, 1, 0])
                self.data += [Image.fromarray(img)]
                self.labels += [label]

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __getitem__(self, index):
        img = self.transform(self.data[index])
        return img, self.labels[index]

    def __len__(self):
        return len(self.data)
