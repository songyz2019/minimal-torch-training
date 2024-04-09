import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn.functional import cross_entropy
from model import MyModel

# dataset
dataset = torchvision.datasets.MNIST(train=True, download=True, root='./data', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model and Optimizer
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(50):
    for x, y in dataloader:
        # Move x and y to GPU
        x, y = x.cuda(), y.cuda()

        # Forward
        optimizer.zero_grad()
        y_hat = model(x)
        loss = cross_entropy(y_hat, y)

        # Backward
        loss.backward()
        optimizer.step()

        # Log
        print("epoch = %d, loss = %f" % (epoch, loss))

# Save model
torch.save(model.state_dict(), 'model.pt')
