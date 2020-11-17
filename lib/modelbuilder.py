import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Build a custom deep neural network model

    # Arguments
        None

    """

    def __init__(self):
        super(Model, self).__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 16, (3,3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3,3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, (3,3), padding=1)
        # pooling layer
        self.pool = nn.MaxPool2d((2,2), 2)
        
        # dense layer
        self.fc1 = nn.Linear(64*8*8, 128)
        # classification/output layer
        self.fc2 = nn.Linear(128, 3)
    
    """
    Forward propagation method

    # Arguments
        x: input (feature vector)
    
    # Output
        x: prediction vector
    """

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x