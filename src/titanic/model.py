import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, n_features) -> None:
        super(LeNet, self).__init__()
        n_fc1 = 128
        n_fc2 = 128
        
        self.fc1 = nn.Linear(n_features, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.fc3 = nn.Linear(n_fc2, 1)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
        