import torch
import gymnasium as gym
import numpy as np
from Actor import Actor

env = gym.make("Pendulum-v1",
               render_mode="human",
              )
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy_net = Actor(3, 1, 2)
policy_net.load_state_dict(torch.load("policy_net_model.pth", map_location=torch.device('cpu')))
policy_net.eval()  # Set the network to evaluation mode

with torch.no_grad():  # Disable gradient calculation
    for _ in range(10):  # Run for 10 episodes
        state = env.reset()
        state = np.array(state[0])
        done = False
        while not done:
            state = torch.FloatTensor(state.reshape(1, -1))
            action = policy_net(state).cpu().data.numpy().flatten()
            next_state, reward, done, _, _ = env.step(action)
            env.render()  # Comment this out if you don't want to see the simulation
            state = next_state

env.close()