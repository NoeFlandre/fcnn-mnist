from torchvision import datasets, transforms
from torch import nn
import torch
from torch.utils.data import DataLoader
from train_loop import train_loop
from test_loop import test_loop
from model import FCNNMNIST

INPUT_SIZE = 28*28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = FCNNMNIST(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

for t in range(EPOCHS):
    print(f"Epoch : {t}")
    train_loop(optimizer=optimizer, loss_fn=loss_fn, loader=train_dataloader, model=model, device=device)
    test_loop(loss_fn=loss_fn, loader=test_dataloader, model=model, device=device)