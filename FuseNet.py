from common import *

class FuseNet(nn.Module):
    def __init__(self, in_shape=(3 * 256,), num_classes=12, mode='logits'):
        super(FuseNet, self).__init__()
        self.mode = mode
        in_channels = in_shape[0]

        self.linear1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.fc(x)
        x = F.relu(x, inplace=True)

        x = F.dropout(x, p=0.5, training=self.training)
        
        return x
