from PIL import Image
import cv2
import glob
import os

import torch
from torch import nn
import torchvision.models as models
from torchinfo import summary

class CombinationModel(nn.Module):
  def __init__(self, num_class=15):
    super().__init__()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = models.ResNet50_Weights.IMAGENET1K_V1

    self.model = models.resnet50(weights=weights).to(device)

    for param in self.model.parameters():
      param.requires_grad = False

    unfrozen_layers = ['fc', 'layer4.2', 'layer4.1']
    for name, param in self.model.named_parameters():
      if any(layer_name in name for layer_name in unfrozen_layers):
        param.requires_grad = True
    
    linear = torch.nn.Linear(1000, 768)
    self.model = torch.nn.Sequential(self.model, linear)

    self.fc1 = nn.Linear(in_features=1068, out_features=768)
    self.fc2 = nn.Linear(in_features=768, out_features=num_class)

  def forward(self, input, text_embedding):
    output = self.model(input)
    new_output = torch.cat((output, text_embedding), dim=1)
    new_output = self.fc1(new_output)
    new_output = self.fc2(new_output)
    return new_output
  
if __name__ == '__main__':



    model = CombinationModel(15)


    # for name, param in model.named_parameters():
    #     print(name)
        # param.requires_grad = True

    summary(model, [(1, 3, 256, 256), (1, 300)])

    # print(model)