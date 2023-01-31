import sys
sys.path.append('./')
from ENV import Env
from main import Agent
from buffer.buffer import Buffer

buffer = Buffer()



if __name__ == "__main__":
    agent = Agent(16,8,0.0009,0.0005,buffer)    
    agent.run(1000000,350)
    episodes = "1M"
    batch_size = "350"