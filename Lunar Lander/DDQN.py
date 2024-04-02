import torch
import torch.nn as nn
import torch.optim as optim
from DQN import DQN
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make(                          ## initialize the enverionment
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

policy_net = DQN(8, 4).to("cuda")
#policy_net.load_state_dict(torch.load('policy_net_model.pth'))

target_net = DQN(8, 4).to("cuda")
target_net.load_state_dict(policy_net.state_dict())          # initialize the weights and bias
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001, weight_decay=1e-5)
replay_memory = []
criterion = nn.MSELoss()

batch_size = 64
gamma = 0.99
epsilon = 1.0
max_steps = 900
episode_rewards = []

for episode in range(1200):
    observation = env.reset()  # Get a new environment
    observation = observation[0]  # Get the valid information ( filter the empty element in the tuple)
    done = False  # If landed
    total_reward = 0
    step_count = 0  # Count the step
    hover_counter = 0
    last_position = None  # track the previous position
    total_loss = 0

    while not done:

        if step_count >= max_steps:  # Check for maximum steps
            break

        current_position = (observation[0], observation[1])

        if last_position is not None:  # Avoid the lander from hovering too long
            distance = np.sqrt(
                (current_position[0] - last_position[0]) ** 2 + (current_position[1] - last_position[1]) ** 2)
            if distance < 0.05:
                hover_counter += 1
            else:
                hover_counter = 0  # Reset the counter if the lander has moved

        last_position = current_position

        state_tensor = torch.FloatTensor(observation).unsqueeze(0).to("cuda")

        # Epsilon-greedy action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Which action the lander takes, 0,1,2,3
        else:
            with torch.no_grad():  # disable the gradient tracking
                action = policy_net(state_tensor).max(1)[1].item()  # get the best action

        next_observation, reward, done, truncated, info = env.step(action)

        if hover_counter >= 90:  # If the lander hovers for 40 steps
            reward -= 0.2  # Apply a penalty of 10 to the reward

        # Store transition
        replay_memory.append((observation, action, reward, next_observation, done))
        if len(replay_memory) > 100000:
            replay_memory.pop(0)

        # Sample mini-batch and update policy_net
        if len(replay_memory) >= batch_size:  # When the length of buffer greater than the batch size
            batch = random.sample(replay_memory, batch_size)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

            batch_states = torch.FloatTensor(batch_states).to("cuda")
            batch_actions = torch.LongTensor(batch_actions).to("cuda")
            batch_rewards = torch.FloatTensor(batch_rewards).to("cuda")
            batch_next_states = torch.FloatTensor(batch_next_states).to("cuda")
            batch_dones = torch.FloatTensor(batch_dones).to("cuda")

            # The TD algo
            current_q_values = policy_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze()
            next_state_actions = policy_net(batch_next_states).max(1)[1]
            next_q_values = target_net(batch_next_states).gather(1, next_state_actions.unsqueeze(1)).squeeze()
            expected_q_values = batch_rewards + gamma * next_q_values * (1 - batch_dones)

            loss = criterion(current_q_values, expected_q_values.detach())
            total_loss += loss.item()  # Accumulate the loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1

            if (step_count % 20 == 0):
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(
                        0.25 * policy_param.data + 0.75 * target_param.data)  # update the weights and bias

        observation = next_observation
        total_reward += reward
    episode_rewards.append(total_reward)

    # Decay epsilon
    if epsilon > 0.1:
        epsilon *= 0.995

    average_loss = total_loss / step_count if step_count > 0 else 0

    print(f"Episode {episode}: Total Reward = {total_reward}, Average Loss = {average_loss:.4f}, Epsilon: {epsilon}")

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