#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class MNISTBody(nn.Module):
    '''
    A CNN with ReLU activations.
    
    Input shape:    (batch_size, 28, 28)
    Output shape:   (batch_size, 50)
    '''
    
    def __init__(self):
        
        super(MNISTBody, self).__init__()
        
        same_padding = (5 - 1) // 2
        
        self.conv1 = nn.Conv2d(1, 10, 5, padding=same_padding)
        self.conv2 = nn.Conv2d(10, 10, 5, padding=same_padding)
        self.lin1  = nn.Linear(10 * 7 * 7, 50)
        self.feature_dim = 50
    
    def forward(self, x):
    
        x = x[:, None, :, :]
    
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2)(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2)(x)
        
        x = x.view(-1, 10 * 7 * 7)
        x = self.lin1(x)
        
        return x
        
class MNISTBodyPaper(nn.Module):
    '''
    A CNN with ReLU activations. Replicates MNIST architecture from
        https://arxiv.org/pdf/1811.06032.pdf
    
    Input shape:    (batch_size, 28, 28)
    Output shape:   (batch_size, 50)
    '''
    
    def __init__(self):
        
        super(MNISTBodyPaper, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 10, 5)
        self.lin1  = nn.Linear(360, 512)
        self.feature_dim = 512
    
    def forward(self, x):
    
        x = x[:, None, :, :]
    
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(-1, 360)
        x = self.lin1(x)
        
        return x