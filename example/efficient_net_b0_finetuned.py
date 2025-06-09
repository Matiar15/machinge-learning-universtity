import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0FineTuned(nn.Module):
    def __init__(self):
        super(EfficientNetB0FineTuned, self).__init__()

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 10)


    def forward(self, x):
        return self.model(x)
