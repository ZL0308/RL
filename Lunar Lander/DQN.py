import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Define the first FC layer
        self.fc1 = nn.Linear(input_dim, 256)
        # Define the second FC layer
        self.fc2 = nn.Linear(256, 128)
        # Define the third FC layer
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        #x = nn.Dropout(p=0.5)(x)
        
        x = self.fc2(x)
        x = nn.ReLU()(x)
        #x = nn.Dropout(p=0.5)(x)
        
        x = self.fc3(x)
        return x
