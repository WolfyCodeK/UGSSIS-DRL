import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import src.config as config

"""
Implementation referenced from:
https://github.com/udacity/deep-reinforcement-learning
"""

def convert_to_tensor(state, device):
    if isinstance(state, np.ndarray):
        state_tensor = torch.from_numpy(state).float().to(device)
    elif isinstance(state, list):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    else: 
        state_tensor = state.to(device)

    state_tensor = state_tensor.view(-1) # Ensure 1D
    state_tensor = state_tensor.unsqueeze(0) # Add batch dimension

    return state_tensor

# MLP to predict Q-value/expected reward from global state
class QNetwork(nn.Module):
    def __init__(self, state_size=config.DQN_STATE_SIZE, action_size=1, hidden_sizes=config.DQN_HIDDEN_SIZES):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        input_size = state_size
        
        # Input layer and hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())
            input_size = hidden_size
            
        # Output layer
        self.layers.append(nn.Linear(input_size, action_size))

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        return x

# Replay buffer
Experience = namedtuple("Experience", field_names=["state", "reward"])

# Fixed-size buffer to store experience tuples
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_size, device):
        self.memory = deque(maxlen=buffer_size)
        
        self.batch_size = batch_size
        self.state_size = state_size
        self.device = device

    def add(self, state, reward):
        state_tensor = convert_to_tensor(state, self.device)

        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        e = Experience(state_tensor, reward_tensor)
        
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert experiences to tensors and add to batch dimension
        states = torch.cat([e.state for e in experiences if e is not None], dim=0).to(self.device)
        rewards = torch.cat([e.reward for e in experiences if e is not None]).to(self.device)

        return (states, rewards)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, device):
        self.state_size = config.DQN_STATE_SIZE
        self.device = device
        self.t_step = 0
        self.min_q_observed = float('inf')
        self.max_q_observed = float('-inf')

        # Initialize networks
        self.qnetwork_local = QNetwork(self.state_size).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.DQN_LR)
        
        # Initial full update
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0) 

        # Replay memory
        self.memory = ReplayBuffer(config.DQN_BUFFER_SIZE, config.DQN_BATCH_SIZE, self.state_size, self.device)

    def get_q_value(self, state):
        state_tensor = convert_to_tensor(state, self.device)
        
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            q_value = self.qnetwork_local(state_tensor).item()
            
        self.qnetwork_local.train()
        
        self.update_q_range(q_value)
        
        return q_value

    def step(self, state, reward):
        # Save experience and learn based on global state and reward
        self.memory.add(state, reward)

        # Learn every set number of time steps       
        if (self.t_step + 1) % config.DQN_UPDATE_EVERY == 0:
            
            if len(self.memory) > config.DQN_BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)
                
            self.soft_update(self.qnetwork_local, self.qnetwork_target, config.DQN_TAU)

        self.t_step += 1 

    def learn(self, experiences):
        # Update value parameters using given batch of experience tuples
        states, rewards = experiences

        # Get predicted Q values from local model
        q_predicted = self.qnetwork_local(states)

        # Calculate and minimize loss
        loss = F.mse_loss(q_predicted, rewards)
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent large updates
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0) # Clip norm to 1.0
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_q_range(self, q_value):
        self.min_q_observed = min(self.min_q_observed, q_value)
        self.max_q_observed = max(self.max_q_observed, q_value)

    def get_q_range(self):
        min_q = self.min_q_observed if self.min_q_observed != float('inf') else 0.0
        max_q = self.max_q_observed if self.max_q_observed != float('-inf') else 0.0
        if max_q <= min_q:
            max_q = min_q + 1e-6 
        return min_q, max_q