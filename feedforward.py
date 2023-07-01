import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from visualize import plot_loss_per_steps

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 784 #28x28
hidden_size = 100
num_classes = 10 #digits from 0 to 9
num_epochs = 5
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



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        #create a Sequential container
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.feedforward(x)

model = NeuralNet(input_size=input_size,
                  hidden_size=hidden_size,
                  num_classes=num_classes)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=learning_rate)

#training loop
training_loss_per_step = []
training_steps = []
n_total_steps = len(train_loader)
start = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print(f"epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")
            training_loss_per_step.append(loss.item())
            training_steps.append(start + i + 1)
    start += 600

#plot loss
plot_loss_per_steps(loss=training_loss_per_step, steps=training_steps)

#test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"accuracy = {acc}")


