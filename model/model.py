import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class MyEfficientNet(nn.Module):

    def __init__(self):
        super().__init__()

        # EfficientNet
        self.network = EfficientNet.from_pretrained("efficientnet-b0")
        # Replace last layer
        self.network._fc = nn.Linear(self.network._fc.in_features, 512)
        self.classifier = nn.Linear(512,30)
    def forward(self, x):
        out = self.network(x)
        return out
