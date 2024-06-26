import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.current_capacity = 0
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.current_capacity += 1
    
    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return random.sample(self.buffer, len(self.buffer))
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
