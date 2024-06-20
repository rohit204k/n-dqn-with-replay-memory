import random
import torch
import torch.nn as nn
import torch.optim as optim

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQN():
    def __init__(self, input_dim, output_dim, env_action_space, epsilon) -> None:
        # Network related initializations
        self.q_function = NN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_function.parameters())
        self.target_q_function = NN(input_dim, output_dim)
        self.target_q_function.load_state_dict(self.q_function.state_dict())

        # Environment related initializations
        self.episodes_ran = 0
        self.accumulated_rewards = []
        self.action_space = env_action_space

        # Hyperparameters 
        self.epsilon = epsilon

    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.q_function(torch.tensor(state, dtype=torch.float32)).argmax().item()
        else:
            return self.action_space.sample()

    def train(self, replay_buffer, batch_size, gamma):
        if len(replay_buffer)<batch_size:
            return
        
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_function(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_q_function(next_states).max(1)[0]  ## Will be implemented later
        # next_q_values = self.q_function(next_states).max(1)[0]
        expected_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return

    def evaluate(self, env, num_episodes):
        total_rewards = []

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    action = self.q_function(torch.tensor(state, dtype=torch.float32)).argmax().item()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        return total_rewards