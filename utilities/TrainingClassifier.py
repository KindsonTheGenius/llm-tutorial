
#Import the required modules
import torch
import torchvision
import torchvision.transforms as transforms
from sympy.core.random import shuffle
from sympy.physics.control.control_plots import matplotlib

from main import train

# Load the CIFAR10 Dataset
# The images from this datasets are PILImages of range[0,1]. We would need to transform them to Tensors of normalized range [-1.1]
transform = transforms.Compose(
    [transforms.ToTensor(), # Converts a PILImage into a PyTorch Tensor
     transforms.Normalize(0.0,0.5,0.5), (0.5, 0.5, 0,5)] # Mean of 0 and SD of 1
)

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)


testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Display some of the images
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # denormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


if __name__ == "__main__":
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show the images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
