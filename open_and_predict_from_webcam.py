from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from torchvision import transforms, models
from utils import Model, train

import numpy as np
import argparse
import pickle
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-p', '--path', required=True, help='Path to the embedded dataset') 
arg_parser.add_argument('-d', '--detector', required=True, help='Path to face detector file')
arg_parser.add_argument('-m', '--recognition_model', required=True, help='Path to face recognition file')
arg_parser.add_argument('-r', '--resize', required=True, help='Resize face to a certain size (must be the same size with the one you specified in extract_dataset.py)')
arg_parser.add_argument('-s', '--split_value', required=True, help='Train/test data split value (decimal point, e.g. 0.8)')
arg_parser.add_argument('-b', '--batch_size', required=True, help='Train/test batch size')
arg_parser.add_argument('-e', '--epochs', required=True, help='Number of training epochs')
arg_parser.add_argument('-l', '--learning_rate', required=True, help='Training learning rate')
args = vars(arg_parser.parse_args())

# print(args['path'])
# print(args['detector'])
# print(args['recognition_model'])
# print(args['resize'])
# print(args['split_value'])
# print(args['batch_size'])
# print(args['epochs'])
# print(args['learning_rate'])

with open('face_mask_dataset.pickle', 'rb') as f:
    loaded_data = pickle.load(f)

features = loaded_data['features']
labels = loaded_data['labels']

IMG_SIZE = int(args['resize'])
split_val = float(args['split_value'])

features = np.asarray(features)
features = np.reshape(features, [len(features), 3, IMG_SIZE, IMG_SIZE])
labels = np.asarray(labels)
n_train = int(split_val * len(features))

train_features = features[:n_train]
train_labels = labels[:n_train]
test_features = features[n_train:]
test_labels = labels[n_train:]

train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
test_data = torch.utils.data.TensorDataset(torch.from_numpy(test_features), torch.from_numpy(test_labels))

BATCH_SIZE = int(args['batch_size'])
NUM_WORKERS = 0
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

model = Model()
print(model)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Training using GPU:', torch.cuda.get_device_name(0))
    model = model.cuda()
else:
    print('No Nvidia GPU found, training using CPU')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(args['learning_rate']))

final_model, training_losses = train(int(args['epochs']), model, train_loader, criterion, optimizer, use_cuda, save_path='')

