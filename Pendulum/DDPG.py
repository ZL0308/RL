import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from Critic import Critic
from Actor import Actor
from Buffer import ReplayBuffer
import matplotlib.pyplot as plt

policy_net = Actor(3,1,2).to("cuda")   # makes the decision
policy_target = Actor(3,1,2).to("cuda")

value_net = Critic(3,1).to("cuda")
value_target = Critic(3,1).to("cuda")

actor_optimizer = optim.Adam(policy_net.parameters())
critic_optimizer = optim.Adam(value_net.parameters())
env = gym.make('Pendulum-v1', g=9.81)
policy_target.load_state_dict(policy_net.state_dict())
value_target.load_state_dict(value_net.state_dict())
replay_buffer = ReplayBuffer()

tau = 0.005
discount = 0.99
batch_size = 64
episode_num = 500
step_per_episode = 200
expl_noise = 0.1
episode_rewards=[]

for episode in range(episode_num):
    state = env.reset()
    state = np.array(state[0])
    total_reward = 0
    transition_count = 0

    for step in range(step_per_episode):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to("cuda")  # reshape the array into one row
        action = policy_net(state_tensor).cpu().data.numpy().flatten()
        noise = np.random.normal(0, expl_noise, size=env.action_space.shape[
            0])  # Added some noise to avoid from trapping into sub-optimal
        action = (action + noise).clip(env.action_space.low,
                                       env.action_space.high)  # Guarantee the value inside of the boundary
        transition_count += 1

        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add((state, next_state, action, reward, done))

        if len(replay_buffer.storage) > batch_size:
            # Sample replay buffer
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            state_batch = torch.FloatTensor(batch_states).to("cuda")
            next_state_batch = torch.FloatTensor(batch_next_states).to("cuda")
            action_batch = torch.FloatTensor(batch_actions).to("cuda")
            reward_batch = torch.FloatTensor(batch_rewards).unsqueeze(1).to("cuda")
            done_batch = torch.FloatTensor(batch_dones).unsqueeze(1).to("cuda")

            # Compute the target Q value
            target_Q = value_target(next_state_batch, policy_target(next_state_batch)).to("cuda")
            target_Q = reward_batch + ((1 - done_batch) * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = value_net(state_batch, action_batch)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q, target_Q)

            # Optimize the critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Compute actor loss
            actor_loss = -value_net(state_batch, policy_net(state_batch)).mean()

            # Optimize the actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if transition_count % 20 == 0:
                # Update the frozen target models
                for param, target_param in zip(value_net.parameters(), value_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(policy_net.parameters(), policy_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode: {episode + 1}, Reward: {total_reward}")

plt.figure(figsize=(10, 5))
plt.plot(range(len(episode_rewards)), episode_rewards, label='Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.savefig('episode_rewards1.png')
plt.show()

env.close()
torch.save(policy_net.state_dict(), 'policy_net_model1.pth')