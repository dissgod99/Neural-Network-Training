import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 784 #28x28
hidden_size = 100
num_classes = 10 #digits from 0 to 9
num_epochs = 2
batch_size = 100
learning_rate = 0.001


#MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", 
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root="./data", 
                                          train=False, 
                                          transform=transforms.ToTensor())

#load datasets and make thenm iterable
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# Iterate over the DataLoader to display examples
for images, labels in train_loader:
    # Display each image in the batch
    print(f"Size images == {images.shape}")
    print(f"Size labels == {labels.shape}")
    for i in range(len(images)):
        image = images[i].squeeze().numpy()  # Convert tensor to numpy array
        label = labels[i].item()  # Get the label value

        # Plot the image
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()
        break  # Stop after the first batch (you can remove this line to display more examples)

    break  # Stop after the first batch (you can remove this line to display more examples)





