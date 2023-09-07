import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.autograd.variable as v
import numpy as np
import cv2
sys.path.append('./')
from ENV import Env
import pybullet as p
from neural_network import Actor,Critic

class Agent:
    def __init__(self,state_size,action_size,lr1,lr2,buffer):
        self.state_size = state_size 
        self.action_size = action_size
        self.lr1 = lr1
        self.lr2 = lr2
        self.gamma = 0.99
        self.lamda = 0.95
        self.buffer = buffer
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.state_size,self.action_size).to(self.device)
        self.critic= Critic(self.state_size,self.action_size).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters() ,lr = self.lr1)
        self.critic_optim = optim.Adam(self.critic.parameters() ,lr = self.lr2)
    def choose_action(self,state):
        image, data = state
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape(3,512,450)
        image = torch.tensor(image,dtype= torch.float32).to(self.device)
        data  = torch.tensor(data,dtype = torch.float32).to(self.device)  
        action =self.actor([image, data])
        
        return action
    def q_value(self,state,action):
        image, data = state
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape(3,240,240)
        image = torch.tensor(image,dtype= torch.float32).to(self.device)
        data  = torch.tensor(data,dtype = torch.float32).to(self.device)
        state = [image,data]
        value =self.critic(state,action)
        return value
    def  discounted_reward(self,reward,done,value,next_value):
        gae = 0
        returns = []
        delta = reward + self.gamma*(1-done)*next_value - value
        gae   = delta + self.gamma*self.lamda*delta
        for i in range(8):
            returns.append(gae)
        return returns
    def image_preprocess(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image
    def loadandsave(self):
        torch.save(self.actor.state_dict() , "weights/actor.pth")
        torch.save(self.critic.state_dict() , "weights/critic.pth")
        torch.save(self.actor_optim.state_dict(),"weights/actor_optim.pth")
        torch.save(self.critic_optim.state_dict(),"weights/critic_optim.pth")
    def learn(self,state,next_state,reward,done,action,inf,speed):
        reward   = torch.tensor(reward ,dtype = torch.float32).to(self.device)
        done     = torch.tensor(done,dtype = torch.float32).to(self.device)
        
        #action   = self.choose_action(state).to(self.device)
        next_action =  self.choose_action(next_state).to(self.device)

        value   = self.q_value(state,action).to(self.device)
        next_value = self.q_value(next_state,next_action).to(self.device)

        advantage = self.discounted_reward(reward,done,value,next_value)
        
        
        log_prob =  f.softmax(action)
        next_log_prob = f.softmax(next_action)
        ratio = next_log_prob/log_prob   
           
        
        s1    = ratio *advantage[0]
        s2    = torch.clamp(ratio,0.8,1.2)
        
        actor_los = torch.min(s1,s2)
        critic_loss = (advantage[0] - value)**2
        critic_loss = critic_loss.mean()
        loss = actor_los + 0.5 *critic_loss
        loss = torch.tensor(loss.mean(),requires_grad= True)

        self.buffer.reward.append(reward.detach().numpy())
        self.buffer.loss.append(loss.detach().numpy())
        self.buffer.values.append(value.detach().numpy())
        self.buffer.next_values.append(next_value.detach().numpy())
        self.buffer.speed.append(speed)
        self.loadandsave()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
    def ploting(self):
        self.buffer.reward_plot()
        self.buffer.loss_plt()
        self.buffer.values_plt()
        self.buffer.speed_plt()
        self.buffer.nvvalues_plt()
    def run(self,episodes,steps):
        count = 0
        for  i in range(episodes):
            env = Env()
            env.simulate()
            state = env.reset()            
            for j in range(steps):                
                action = self.choose_action(state)                
                next_state,reward,done,info,speed = env.step(action)
                count = count + 1
                self.buffer.count.append(count)
                if done:
                    self.learn(state,next_state,reward,done,info,speed)
                    self.ploting()
                    state = next_state
                    print("+++++++++++++++++++++++++++++++")
                else:
                    self.learn(state,next_state,reward,done,action,info,speed)
                    state = next_state
                    self.ploting()
                    print("======================================================================================================")
                    print("episodes:",count,"/",episodes*steps,",","reward:",reward)
                    print("angles",info)
                    print("------------------------------------------------------------------------------------------")
            env.close()
