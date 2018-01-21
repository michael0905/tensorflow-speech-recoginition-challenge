from common import*

#https://www.kaggle.com/alphasis/light-weight-cnn-lb-0-74

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



class VggNet1(nn.Module):
    def __init__(self, in_shape=(1,40,101), num_classes=12, mode='logits' ):
        super(VggNet1, self).__init__()
        self.mode = mode

        self.conv1a = ConvBn2d( 1,  8, kernel_size=(3, 3), stride=(1, 1))
        self.conv1b = ConvBn2d( 8,  8, kernel_size=(3, 3), stride=(1, 1))
        self.conv2a = ConvBn2d( 8, 16, kernel_size=(3, 3), stride=(1, 1))
        self.conv2b = ConvBn2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
        self.conv3a = ConvBn2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv3b = ConvBn2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.linear1 = nn.Linear(32*12*5,512)
        self.linear2 = nn.Linear(512,256)
        self.fc     = nn.Linear(256,num_classes)



    def forward(self, x):

        x = F.relu(self.conv1a(x),inplace=True)
        x = F.relu(self.conv1b(x),inplace=True)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.conv2a(x),inplace=True)
        x = F.relu(self.conv2b(x),inplace=True)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.conv3a(x),inplace=True)
        x = F.relu(self.conv3b(x),inplace=True)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x),inplace=True)

        if self.mode == 'logits':
            x = F.dropout(x,p=0.2,training=self.training)
            x = F.relu(self.linear2(x),inplace=True)
            x = self.fc(x)
            return x  #logits
        elif self.mode == 'features':
            return x


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


    net = VggNet1(in_shape=(1,height,width), num_classes=num_classes)
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


