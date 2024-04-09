from torch.nn import Module, Sequential, Linear, Conv2d, ReLU, MaxPool2d, Flatten

class MyModel(Module):
    def __init__(self, in_channels=1):
        super(MyModel, self).__init__()
        self.cnn = Sequential(
            Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2)
        )
        self.classifier = Sequential(
            Flatten(),
            Linear(7*7*64, 128),
            ReLU(),
            Linear(128, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        y = self.classifier(x)
        return y