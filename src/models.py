import torch.nn as nn


class SRCNN1(nn.Module):
    def __init__(self):
        super(SRCNN1, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=9, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 64, kernel_size=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.layers(x)


class SRCNN2(nn.Module):
    def __init__(self):
        super(SRCNN2, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=9, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 32, kernel_size=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 3, kernel_size=5, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


class SRCNN3(nn.Module):
    def __init__(self):
        super(SRCNN3, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=9, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 64, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 32, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 16, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(16, 3, kernel_size=5, padding=1)
        )

    def forward(self, x):
        return self.layers(x)
