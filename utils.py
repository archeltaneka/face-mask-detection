import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, (3,3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3,3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.pool = nn.MaxPool2d((2,2), 2)
        
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 3)
        
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

def train(num_epochs, model, loader, crit, opt, use_cuda, plot_curve=True, save_path=''):
    
    training_losses = []
    
    for i in range(num_epochs):
        training_loss = 0.0
        
        model.train()
        for batch, (feature, label) in enumerate(loader):
            feature = feature.to(torch.float)
            label = label.to(torch.long)
            if use_cuda:
                feature, label = feature.cuda(), label.cuda()
            
            opt.zero_grad()
            
            output = model(feature)
            loss = crit(output, label)
            loss.backward()
            opt.step()
            
            training_loss = training_loss + (1 / (batch + 1)) * (loss.data - training_loss)
            
        print("Epoch #{} | Training loss: {}".format(i+1, training_loss))
        training_losses.append(training_loss)
    
    if len(save_path) > 0:
        print('Saving model...')
        torch.save(model.state_dict(), save_path)
        print('Model saved:', save_path)
        
    if plot_curve:
        plt.plot(range(num_epochs), training_losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
    return model, training_losses