import torch
import torch.nn as nn

from torchvision.models import swin_v2_b

# Base version of the classifier which only uses the Light curve image
class LC_classifier(nn.Module):

    def __init__(self):
        super(LC_classifier, self).__init__()
        self.swin = swin_v2_b()
    
    def forward(self, x):
        return self.swin(x)


# Secondary version of the classifier which only uses both the Light curve image and the postage stamp image
class LC_classifier_MM(nn.Module):

    def __init__(self):
        super(LC_classifier_MM, self).__init__()
        self.swin = swin_v2_b()
        self.fc = nn.Linear(1000, 1)
    
    def forward(self, x):
        x = self.swin(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    
    model = LC_classifier()
    print(model(torch.rand(1, 3, 256, 256)).shape)