
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dqn import DQN
from replay_buffer import ReplayBuffer


import gymnasium as gym

env = gym.make('CartPole-v1')


input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

replay_buffer = ReplayBuffer(2500) # 10000

def gen_q_function(n, input_dim, output_dim):
    q_functions = []
    target_q_functions = []
    optimizers = []

    for _ in range(n):

        q_function = DQN(input_dim, output_dim)
        q_functions.append[q_function]

        # target_q_function = DQN(input_dim, output_dim)
        # target_q_function.load_state_dict(q_function.state_dict())
        # target_q_functions.append(target_q_function)

        optimizer = optim.Adam(q_function.parameters())
        optimizers.append(optimizer)


        return q_function, target_q_functions, optimizers


def select_action(agent, state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return agent(torch.tensor(state, dtype=torch.float32)).argmax().item()
    else:
        return env.action_space.sample()
    
def train(agent, target_agent, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) <= batch_size:
        return
    
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # next_q_values = target_agent(next_states).max(1)[0]  ## Will be implemented later
    next_q_values = agent(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate_agent(agent, env, num_episodes=10):
    total_rewards = []
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                action = agent(torch.tensor(state, dtype=torch.float32)).argmax().item()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return total_rewards


# def plot_results(agent1_rewards_over_time, agent2_rewards_over_time):
    
#     # Assuming you have lists to track total rewards per episode for both agents during training
#     agent1_rewards_over_time = []
#     agent2_rewards_over_time = []

#     # After the training loop, append the total rewards of each episode to these lists
#     # For demonstration purposes, let's assume you have these lists already filled

#     plt.figure(figsize=(12, 6))
#     plt.plot(agent1_rewards_over_time, label='Agent 1')
#     # plt.plot(agent2_rewards_over_time, label='Agent 2')
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.title('Total Rewards per Episode')
#     plt.legend()
#     plt.show()

def main():
    num_episodes = 10000
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    target_update = 10

    no_agents = 2
    q_functions, target_q_functions, optimizers = gen_q_function(no_agents, input_dim, output_dim)

    num_eval_episodes = 10
    eval_interval = 50

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        
        index = random.randint(0, no_agents - 1)

        while not done:
            action = select_action(q_functions[index], state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            
            train(q_functions[index], target_q_functions[index], optimizers[index], replay_buffer, batch_size, gamma)
            
            if done:
                break
        
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        
        # if episode % target_update == 0:
            # target_agent1.load_state_dict(agent1.state_dict())  ## Will be implemented later
        
        # if episode % eval_interval == 0:
        #     agent_rewards = evaluate_agent(q_functions[index], env, num_eval_episodes)
        #     agent_mean_reward = np.mean(agent_rewards)
        #     agent_std_reward = np.std(agent_rewards)
            
        #     print(f'Episode {episode}:')
        #     print(f'  Agent 1 Mean Reward: {agent_mean_reward}, Std Dev: {agent_std_reward}')


    # plot_results(agent1_rewards_over_time, [])
        

if __name__ == "__main__":
    main()