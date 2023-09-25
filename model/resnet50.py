from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision import datasets, transforms, models
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, num_class):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Linear(256, num_class))
        self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        unfrozen_layers = ['fc', 'layer4.2', 'layer4.1']
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in unfrozen_layers):
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    from torchinfo import summary

    model = ResNet50(15)


    summary(model, (1, 3, 256, 256))






