import matplotlib.pyplot as plt
import numpy as np

class Buffer:
    def __init__(self):
        self.count = []
        self.reward =[]
        self.loss   = []
        self.values = []
        self.next_values = []
        self.speed = []
        self.speedx = []
        self.speedy = []
        self.speedy = []
    def reward_plot(self):
        v1 = np.array(self.reward)
        v2 = np.array(self.count)
        plt.plot(v2,v1)#
        plt.xlable = "count"
        plt.ylable = "reward"
        plt.savefig("buffer/reward.png")
        plt.close()
    def loss_plt(self):
        v1 = np.array(self.loss)
        v2 = np.array(self.count)
        plt.plot(v2,v1)#
        plt.xlable = "count"
        plt.ylable = "loss"
        plt.savefig("buffer/loss.png")
        plt.close()    
    def  values_plt(self):
        v1 = np.array(self.values)
        v2 = np.array(self.count)
        plt.plot(v2,v1)#
        plt.xlable = "count"
        plt.ylable = "vlues"
        plt.savefig("buffer/values.png")
        plt.close()    
    def  nvvalues_plt(self):
        v1 = np.array(self.next_values)
        v2 = np.array(self.count)
        plt.plot(v2,v1)#
        plt.xlable = "count"
        plt.ylable = "next_values"
        plt.savefig("buffer/next_values.png")
        plt.close()    
    def speed_plt(self):
        v1 = np.array(self.speed)
        v2 = np.array(self.count)
        plt.plot(v2,v1)#
        plt.xlable = "count"
        plt.ylable = "speed"
        plt.savefig("buffer/speed.png")
        plt.close()   
        


