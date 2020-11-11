from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def evaluate(model, loader, crit, use_cuda):
    
    all_preds = []
    ground_truths = []
    test_loss = 0.0
    num_correct = 0
    total_data = 0

    model.eval()
    for batch, (feature, label) in enumerate(loader):
        feature = feature.to(torch.float)
        label = label.to(torch.long)
        if use_cuda:
            feature, label = feature.cuda(), label.cuda()
        
        output = model(feature)
        loss = crit(output, label)
        
        test_loss = test_loss + (1/(batch+1)) * (loss.data - test_loss)
        total_data = total_data + feature.size(0)
        
        preds = output.data.max(1)[1]
        num_correct += np.sum(np.squeeze(preds.eq(label.data.view_as(preds))).cpu().numpy())
        
        all_preds.append(preds.cpu().detach().numpy().tolist())
        ground_truths.append(label.cpu().detach().numpy().tolist())
        
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {:.2f}%'.format(num_correct / total_data * 100))
    
    pred_temp = []
    label_temp = []
    
    for pred in all_preds:
        for p in pred:
            pred_temp.append(p)
    for labels in ground_truths:
        for l in labels:
            label_temp.append(l)
    
    print('\nConfusion Matrix')
    print(confusion_matrix(pred_temp, label_temp))
    print('\nClassification Report')
    print(classification_report(pred_temp, label_temp))