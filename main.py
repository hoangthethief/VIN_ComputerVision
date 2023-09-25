from model.caption_vu import CombinationModel
import torch
from torch import nn
from lion_pytorch import Lion
from feeder.feeder_caption import LandscapeDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CombinationModel().to(device)

batch_size = 16

optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
loss_func = nn.CrossEntropyLoss()

metadata_path = 'feeder/train.csv'
test_path = 'feeder/test.csv'

train_dataset = LandscapeDataset(metadata_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = LandscapeDataset(test_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

epochs = 5
test_acc = []

for epoch in range(epochs):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (path, images, captions, labels) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, captions)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))
        if i % 25 == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))

    train_accuracy = train_accuracy / (len(train_loader) * batch_size)
    train_loss = train_loss / (len(train_loader) * batch_size)


    model.eval()
    test_accuracy=0.0
    for i, (path, images, captions, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=images.to(device)
            captions = captions.to(device)
            labels=labels.to(device)

        outputs=model(images, captions)
        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))

    test_accuracy=test_accuracy/(len(test_loader) * batch_size)

    


    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    name = 'checkpoint/best_checkpoint' + str(test_accuracy) + ".model"
    torch.save(model.state_dict(),name)
    test_acc.append(test_accuracy)
print(test_acc)
plt.plot(test_acc)
plt.title("Test accuracy")

plt.show()
