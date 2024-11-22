import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    def __init__(
        self,
        filters=6,        # Base number of filters (first conv layer)
        hidden_dim=128,  # Base size for hidden layers
        dropout=0.2      # Dropout rate
    ):
        super(LeNet, self).__init__()
        
        # Conv layers (second conv layer uses 2.5x filters)
        self.conv1 = nn.Conv2d(3, filters, 5)
        self.conv2 = nn.Conv2d(filters, filters * 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate sizes for fully connected layers
        self.flatten_size = (filters * 2) * 5 * 5
        
        # Fully connected layers (second FC uses 0.7x hidden size)
        self.fc1 = nn.Linear(self.flatten_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim * 0.7))
        self.fc3 = nn.Linear(int(hidden_dim * 0.7), 10)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)        
        x = x.view(-1, self.flatten_size)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)        
        x = self.fc3(x)
        return x