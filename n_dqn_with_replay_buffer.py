
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List

from dqn import DQN
from replay_buffer import ReplayBuffer


import gymnasium as gym

def gen_q_function(n, input_dim, output_dim, env_action_space, epsilon, epsilon_end, epsilon_decay, learning_rate=1e-3, tau = 1)->List[DQN]:
    q_functions = []
    for _ in range(n):

        agent = DQN(input_dim, output_dim, env_action_space, epsilon, epsilon_end, epsilon_decay, learning_rate, tau)
        q_functions.append(agent)

    return q_functions

def plot_learning_curve(q_functions:List[DQN]):
    step = 5

    plt.figure(figsize=(12, 6))
    for index in range(len(q_functions)):
        values = q_functions[index].accumulated_rewards[::step]
        episodes = [step*i for i in range(len(values))]
        plt.plot(episodes, values, label=f'Agent {index + 1}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.legend()
    plt.show()
    return

def main():
    num_episodes = 2000
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_end = 0
    epsilon_decay = 5e-5
    target_update = 5
    learning_rate = 8e-3
    tau = 0.9
    tau_decay = 1e-3
    stop_training_at = 180

    env = gym.make('CartPole-v0')

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    replay_buffer = ReplayBuffer(100000) # 10000

    no_agents = 5
    q_functions = gen_q_function(no_agents, input_dim, output_dim, env.action_space, epsilon, epsilon_end, epsilon_decay, learning_rate, tau)

    num_eval_episodes = 10
    eval_interval = 100

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        
        episode_reward = 0
        index = random.randint(0, no_agents - 1)

        while not done:
            action = q_functions[index].select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            q_functions[index].train(replay_buffer, batch_size, gamma)
            q_functions[index].update_epsilon()

            if done:
                break

        q_functions[index].accumulated_rewards.append(episode_reward)
        q_functions[index].episodes_ran += 1
        
        if episode % target_update == 0 and episode != 0:
            q_functions[index].update_target_q_function()
            q_functions[index].tau = max(0.1, q_functions[index].tau-tau_decay)
            # q_functions[index].target_q_function.load_state_dict(q_functions[index].q_function.state_dict())  ## Will be implemented later
        
        if episode % eval_interval == 0 and episode != 0:
            last_n_episodes_avg_reward = np.mean(q_functions[index].accumulated_rewards[-eval_interval:])
            # agent_rewards = q_functions[index].evaluate(env, num_eval_episodes)
            # agent_mean_reward = np.mean(agent_rewards)
            # agent_std_reward = np.std(agent_rewards)
            
            print('-----------------------------------------------------------')
            print(f'Agent: {index + 1}, Episode: {episode}, Epsilon: {q_functions[index].epsilon}, Tau: {q_functions[index].tau}')
            print(f'Episode {episode}th Reward: {episode_reward}, Last {eval_interval} episodes reward: {last_n_episodes_avg_reward}')
            # print(f'Evaluation Mean Reward: {agent_mean_reward}, Std Dev: {agent_std_reward}')

            if last_n_episodes_avg_reward>stop_training_at:
                break
    
    # print(q_functions[0].evaluate(env, num_eval_episodes))
    plot_learning_curve(q_functions)    

if __name__ == "__main__":
    main()