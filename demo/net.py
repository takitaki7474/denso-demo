import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, out=2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(4*4*64, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, out)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #  32*32*3 -> 32*32*16
        x = F.max_pool2d(x, 2, 2) #  32*32*16 -> 16*16*16
        x = F.relu(self.conv2(x)) #  16*16*16 -> 16*16*32
        x = F.max_pool2d(x, 2, 2) # 16*16*32 -> 8*8*32
        x = F.relu(self.conv3(x)) #  8*8*32 -> 8*8*64
        x = F.max_pool2d(x, 2, 2) #  8*8*64 -> 4*4*64
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
