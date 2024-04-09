import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model import MyModel

# dataset
dataset = torchvision.datasets.MNIST(train=False, download=True, root='./data', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Model
model = MyModel().cuda()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Training
n_correct = 0
n_incorrect = 0
for x, y in dataloader:
    # Move x and y to GPU
    x, y = x.cuda(), y.cuda()

    # Forward
    y_hat = model(x)

    # Log
    if torch.argmax(y) == torch.argmax(y_hat):
        n_correct += 1
    else:
        n_incorrect += 1
    print("correct=%d  incorrect=%d" % (n_correct, n_incorrect))


