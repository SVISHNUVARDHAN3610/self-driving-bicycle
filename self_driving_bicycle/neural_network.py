import torch
import torch.nn as nn
import torch.nn.functional as f

class Actor(nn.Module):
    def __init__(self,state_size,action_size):
        super(Actor,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.conv1 = nn.Conv2d(3,9,kernel_size=3,stride=1,padding=0,bias = True)
        self.conv2 = nn.Conv2d(9,18,kernel_size=3,stride=1,padding=0,bias = True)
        self.conv3 = nn.Conv2d(18,9,kernel_size=3,stride=2,padding=0,bias = True)
        self.conv4 = nn.Conv2d(9,3,kernel_size=3,stride=2,padding=0,bias = True)
        self.max   = nn.MaxPool2d(kernel_size=3,stride=1)
        self.linear1= nn.Linear(self.state_size,32)
        self.linear2= nn.Linear(32,64)
        self.linear3= nn.Linear(64,128)
        self.linear4= nn.Linear(128,512)
        self.linear5= nn.Linear(512,128)
        self.linear6= nn.Linear(128,64)
        self.linear7= nn.Linear(64,16) 
    def forward(self,input):
        x1 = f.leaky_relu(self.conv1(input[0]))
        x1 = self.max(x1)
        x1 = f.leaky_relu(self.conv2(x1))
        x1 = self.max(x1)
        x1 = f.leaky_relu(self.conv3(x1))
        x1 = self.max(x1)
        x1 = f.leaky_relu(self.conv4(x1))
        x1 = self.max(x1)
        x1 = torch.reshape(x1,(-1,))
        
        x2 = f.leaky_relu(self.linear1(input[1]))
        x2 = f.leaky_relu(self.linear2(x2))
        x2 = f.leaky_relu(self.linear3(x2))
        x2 = f.leaky_relu(self.linear4(x2))
        x2 = f.leaky_relu(self.linear5(x2))
        x2 = f.leaky_relu(self.linear6(x2))
        x2 = f.leaky_relu(self.linear7(x2))
        
        x  = torch.cat([x1,x2])
        last = nn.Linear(x.shape[0],self.action_size)
        x  = last(x)
        return x



        
class Critic(nn.Module):
    def __init__(self,state_size,action_size):
        super(Critic,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(64,32,kernel_size=3,stride=1,padding=0)
        self.max   = nn.MaxPool2d(kernel_size=3,stride=1)
        self.linear1= nn.Linear(self.state_size,32)
        self.linear2= nn.Linear(32,64)
        self.linear3= nn.Linear(64,128)
        self.linear4= nn.Linear(128,512)
        self.linear5= nn.Linear(512,128)
        self.linear6= nn.Linear(128,64)
        self.linear7= nn.Linear(64,32) 
        self.act1   = nn.Linear(self.action_size,32)
        self.act2   = nn.Linear(32,64)
        self.act3   = nn.Linear(64,32)
        
    def forward(self,input,action):
        x1 = f.leaky_relu(self.conv1(input[0]))
        x1 = f.leaky_relu(self.conv2(x1))
        x1 = self.max(x1)
        x1 = f.leaky_relu(self.conv3(x1))
        x1 = self.max(x1)
        x1 = torch.reshape(x1,(-1,))
        x2 = f.leaky_relu(self.linear1(input[1]))
        x2 = f.leaky_relu(self.linear2(x2))
        x2 = f.leaky_relu(self.linear3(x2))
        x2 = f.leaky_relu(self.linear4(x2))
        x2 = f.leaky_relu(self.linear5(x2))
        x2 = f.leaky_relu(self.linear6(x2))
        x2 = f.leaky_relu(self.linear7(x2))
        x3 = f.leaky_relu(self.act1(action))
        x3 = f.leaky_relu(self.act2(x3))
        x4 = f.leaky_relu(self.act3(x3))
        x  = torch.cat([x1,x2,x3])
        last = nn.Linear(x.shape[0],1)
        x  = last(x)
        return x
        


        