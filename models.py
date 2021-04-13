import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=100, p=0.5):
        """
        https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Dropout(p, inplace=True),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        for index in [4, 10, 12, 17, 20, 22]:
            nn.init.constant_(self.net[index].bias, 1)

    def forward(self, x):
        return self.net(x)


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AE, self).__init__()
        self.criterion = nn.MSELoss()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        mu, log_var = torch.split(self.encoder(x), 2)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
