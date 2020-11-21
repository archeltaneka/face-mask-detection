import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    """
    --Note--
    Feel free to experiment with the neural network architecture:
        1. Add/decrease the number of layers
        2. Change the number of filter, padding, stride
        3. Add another type of layers (e.g. Batch norm)
    """

    def __init__(self):
        """
        Build/initiate a custom deep neural network model

        # Arguments
            None

        # Outputs
            None

        """
        
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

    def forward(self, x):
        """
        Forward propagation method

        # Arguments
            x: torch.Tensor - input (feature vector)
        
        # Output
            x: torch.Tensor - prediction vector
            
        """

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # flattens the input vector
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x