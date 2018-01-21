from common import*

## block ##-------
class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class SeScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SeScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(ResBlock, self).__init__()
        assert(in_planes==out_planes)

        self.conv_bn1 = ConvBn2d(in_planes,  out_planes, kernel_size=3, padding=1, stride=1)
        self.conv_bn2 = ConvBn2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.scale    = SeScale(out_planes, reduction)

    def forward(self, x):
        z  = F.relu(self.conv_bn1(x),inplace=True)
        z  = self.conv_bn2(z)
        z  = self.scale(z)*z + x
        z  = F.relu(z,inplace=True)
        return z



## net ##-------

class SeResNet4(nn.Module):
    def __init__(self, in_shape=(1,40,101), num_classes=12, mode='logits' ):
        super(SeResNet4, self).__init__()
        self.mode = mode

        in_channels = in_shape[0]

        self.layer1a = ConvBn2d(in_channels, 16, kernel_size=(3, 3), stride=(1, 1))
        self.layer1b = ResBlock( 16, 16)

        self.layer2a = ConvBn2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        self.layer2b = ResBlock(32, 32)
        self.layer2c = ResBlock(32, 32)

        self.layer3a = ConvBn2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.layer3b = ResBlock(64, 64)
        self.layer3c = ResBlock(64, 64)

        self.layer4a = ConvBn2d( 64,128, kernel_size=(3, 3), stride=(1, 1))
        self.layer4b = ResBlock(128,128)
        self.layer4c = ResBlock(128,128)

        self.layer5a = ConvBn2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
        self.layer5b = nn.Linear(256,256)

        self.fc = nn.Linear(256,num_classes)


    def forward(self, x):

        x = F.relu(self.layer1a(x),inplace=True)
        x = self.layer1b(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.1,training=self.training)
        x = F.relu(self.layer2a(x),inplace=True)
        x = self.layer2b(x)
        x = self.layer2c(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer3a(x),inplace=True)
        x = self.layer3b(x)
        x = self.layer3c(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer4a(x),inplace=True)
        x = self.layer4b(x)
        x = self.layer4c(x)

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer5a(x),inplace=True)
        x = F.adaptive_max_pool2d(x,1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer5b(x))



        if self.mode == 'logits':
            x = F.dropout(x,p=0.2,training=self.training)
            x = self.fc(x)
            return x  #logits

        elif self.mode == 'features':
            return x









## check ##############################################################################

def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 32
    num_classes = 12
    height = 40
    width  = 101
    labels = torch.randn(batch_size,num_classes)
    inputs = torch.randn(batch_size,1,height,width)
    y = Variable(labels).cuda()
    x = Variable(inputs).cuda()


    net = SeResNet3(in_shape=(1,height,width), num_classes=num_classes)
    net.cuda()
    net.train()


    logits = net.forward(x)
    probs  = F.softmax(logits, dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    #print(net)
    print('probs')
    print(probs)




########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print('sucess')
