from common import*

#https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/44283

class ConvBn1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm1d(out_channels)

        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x



class Simple1dNet(nn.Module):
    def __init__(self, in_shape=(1,16000), num_classes=12, mode='logits' ):
        super(Simple1dNet, self).__init__()
        self.mode = mode

        self.conv1a = ConvBn1d(  1,  8, kernel_size=3, stride=1)
        self.conv1b = ConvBn1d(  8,  8, kernel_size=3, stride=1)

        self.conv2a = ConvBn1d(  8, 16, kernel_size=3, stride=1)
        self.conv2b = ConvBn1d( 16, 16, kernel_size=3, stride=1)

        self.conv3a = ConvBn1d( 16, 32, kernel_size=3, stride=1)
        self.conv3b = ConvBn1d( 32, 32, kernel_size=3, stride=1)

        self.conv4a = ConvBn1d( 32, 64, kernel_size=3, stride=1)
        self.conv4b = ConvBn1d( 64, 64, kernel_size=3, stride=1)

        self.conv5a = ConvBn1d( 64,128, kernel_size=3, stride=1)
        self.conv5b = ConvBn1d(128,128, kernel_size=3, stride=1)

        self.conv6a = ConvBn1d(128,256, kernel_size=3, stride=1)
        self.conv6b = ConvBn1d(256,256, kernel_size=3, stride=1)

        self.conv7a = ConvBn1d(256,256, kernel_size=3, stride=1)
        self.conv7b = ConvBn1d(256,256, kernel_size=3, stride=1)

        self.conv8a = ConvBn1d(256,512, kernel_size=3, stride=1)
        self.conv8b = ConvBn1d(512,512, kernel_size=3, stride=1)

        self.conv9a = ConvBn1d(512,512, kernel_size=3, stride=1)
        self.conv9b = ConvBn1d(512,512, kernel_size=3, stride=1)
        #self.linear1 = nn.Linear(512*31,1024)

        self.conv10a = ConvBn1d( 512,1024, kernel_size=3, stride=1)
        self.conv10b = ConvBn1d(1024,1024, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,256)
        self.fc      = nn.Linear(256,num_classes)



    def forward(self, x):

        #print(x.size())
        x = F.relu(self.conv1a(x),inplace=True)
        x = F.relu(self.conv1b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        #x = F.dropout(x,p=0.10,training=self.training)
        x = F.relu(self.conv2a(x),inplace=True)
        x = F.relu(self.conv2b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        #x = F.dropout(x,p=0.10,training=self.training)
        x = F.relu(self.conv3a(x),inplace=True)
        x = F.relu(self.conv3b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.10,training=self.training)
        x = F.relu(self.conv4a(x),inplace=True)
        x = F.relu(self.conv4b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.10,training=self.training)
        x = F.relu(self.conv5a(x),inplace=True)
        x = F.relu(self.conv5b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv6a(x),inplace=True)
        x = F.relu(self.conv6b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv7a(x),inplace=True)
        x = F.relu(self.conv7b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv8a(x),inplace=True)
        x = F.relu(self.conv8b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv9a(x),inplace=True)
        x = F.relu(self.conv9b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv10a(x),inplace=True)
        x = F.relu(self.conv10b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)
        #------------------------------------------

        #print(x.size())
        x = F.adaptive_avg_pool1d(x,1)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,p=0.50,training=self.training)
        x = F.relu(self.linear1(x),inplace=True)

        if self.mode == 'logits':
            x = F.dropout(x,p=0.50,training=self.training)
            x = F.relu(self.linear2(x),inplace=True)
            x = self.fc(x)
            return x  #logits

        elif self.mode == 'features':
            return x

 


def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 32
    num_classes = 12
    length = 16000

    labels = torch.randn(batch_size,num_classes)
    inputs = torch.randn(batch_size,1,length)
    y = Variable(labels).cuda()
    x = Variable(inputs).cuda()


    net = Simple1dNet_2(in_shape=(1,length), num_classes=num_classes).cuda()
    net.train()

    logits = net.forward(x)
    probs  = F.softmax(logits, dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    #print(net)
    print('probs')
    #print(probs)




########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

