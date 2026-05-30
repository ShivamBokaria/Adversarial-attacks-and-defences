import torchvision.models as models
import torch.nn as nn

def ResNet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    return model