from torch.utils.data import DataLoader, random_split
import zipfile
import torchvision.models as models
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import glob
import os

import torch
from torch import nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.cuda.set_device('cuda:0')
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# print(device)

random_state = 19
img_size = 256
batch_size = 64
data_dir = "/home/thanh/huyendn/CV/proccessed_photo"

transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor()
])

data = datasets.ImageFolder(root=data_dir, transform=transform)
class_to_idx = data.class_to_idx
# print(class_to_idx)
total_dataset_length = len(data)  
train_size = int(0.8 * total_dataset_length) 
test_size = total_dataset_length - train_size

# data_train, data_test = random_split(data, [0.7, 0.3])
data_train, data_test = random_split(data, [train_size, test_size])

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(data_val, batch_size=8, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size)
print(len(train_loader))

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True) 
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

num_classes = 15
model = ResNet101(num_classes).to(device)

# for param in model.parameters():
#     param.requires_grad = False

# unfrozen_layers = []
# for name, param in model.named_parameters():
#     if any(layer_name in name for layer_name in unfrozen_layers):
#         param.requires_grad = True

learning_rate = 0.001

# print(model)
# model = Simple_net(num_classes=15).to(device=device)
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

def test(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation during testing
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Calculate test loss
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_test_loss = test_loss / len(dataloader)
    test_accuracy = 100.0 * correct / total

    return avg_test_loss, test_accuracy

epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Calculate and print average training loss and accuracy for the epoch
    avg_train_loss = train_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    print(f'Epoch [{epoch + 1}/{epochs}] - Train loss: {avg_train_loss:.4f} - Train accuracy: {accuracy:.2f}% - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'resnet101_model.pth')
