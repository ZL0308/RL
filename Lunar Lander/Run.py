import torch
import gymnasium as gym
from DQN import DQN

policy_net = DQN(8, 4)  # Initialize the DQN model (adjust the dimensions if necessary)
policy_net.load_state_dict(torch.load('policy_net_model1.pth', map_location=torch.device('cpu')))
policy_net.eval()

env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode="human",
)

state = env.reset()  # Reset the environment to start
state = state[0]
done = False
total_reward = 0

while not done:
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor
    with torch.no_grad():  # Disable gradient calculations
        action = policy_net(state_tensor).max(1)[1].item()  # Select the action with the highest Q-value

    next_state, reward, done, _, info = env.step(action)  # Take the action in the environment
    total_reward += reward
    state = next_state  # Update the state

print(f"Total reward: {total_reward}")
env.close()