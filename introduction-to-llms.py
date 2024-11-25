#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import click

# Create a click gorup
@click.group()
def cli():
    """Pytorch beginner program ath allows you to change the batch size and epochs."""

@cli.command("train")
@click.option("--batch-size", default=64, help="Batch size for training")
@click.option("--epochs", default=5, help="Number of epochs")
def train(batch_size: int, epochs: int):
    """Train the Neural Network."""
    print(f"Training with batch siz {batch_size} and {epochs} epochs")

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    batch_size = batch_size

    # Crate the Data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y : {y.shape} {y.dtype}")
        break

    # Get cpu or cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Define the model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    print(model)

    # Create a loss function and an optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Train the model
    def train(dataloader, model, loss_function, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # compute the prediction error
            pred = model(X)
            loss = loss_function(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss {loss:>7f}, [{current:5d}/{size:>5d}]")

    # Test the model
    def test(dataloader, model, loss_function):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Loss: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
        train(train_dataloader, model, loss_function, optimizer)
        test(test_dataloader, model, loss_function)
        print("Done!")

if __name__ == "__main__":
    cli()