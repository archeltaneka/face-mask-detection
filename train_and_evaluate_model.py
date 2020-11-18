from torchvision import transforms, models
from lib.utils import train, evaluate
from lib.modelbuilder import Model

import numpy as np
import argparse
import pickle
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print('Initiating model training and evaluation...')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-p', '--path', required=True, help='Path to the embedded dataset') 
arg_parser.add_argument('-s', '--split_value', required=True, help='Train/test data split value (decimal point, e.g. 0.8)')
arg_parser.add_argument('-b', '--batch_size', required=True, help='Train/test batch size')
arg_parser.add_argument('-e', '--epochs', required=True, help='Number of training epochs')
arg_parser.add_argument('-l', '--learning_rate', required=True, help='Training learning rate')
args = vars(arg_parser.parse_args())

print('[INFO] Loading your dataset: {}...'.format(args['path']))

with open(args['path'], 'rb') as f:
    loaded_data = pickle.load(f)

print('[STATUS] Dataset found, loading data completed')
print('[INFO] Converting dataset into tensors...')

features = loaded_data['features']
labels = loaded_data['labels']

IMG_SIZE = 64
print('[INFO] All images will be resized to {}x{}'.format(IMG_SIZE, IMG_SIZE))

split_val = float(args['split_value'])

features = np.asarray(features)
features = np.reshape(features, [len(features), 3, IMG_SIZE, IMG_SIZE])
labels = np.asarray(labels)

print('[INFO] Training/test split: {}/{}'.format(int(split_val*100), int(100 - (split_val*100))))

n_train = int(split_val * len(features))
train_features = features[:n_train]
train_labels = labels[:n_train]
test_features = features[n_train:]
test_labels = labels[n_train:]

print('[INFO] Training data:', len(train_features))
print('[INFO] Testing data:', len(test_features))

train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
test_data = torch.utils.data.TensorDataset(torch.from_numpy(test_features), torch.from_numpy(test_labels))

print('[INFO] Dividing tensors into {} batches'.format(args['batch_size']))

BATCH_SIZE = int(args['batch_size'])
NUM_WORKERS = 0
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

print('[STATUS] Dataset conversion completed')
print('[INFO] Creating a model...')

model = Model()
print(model)

print('[STATUS] Model creation completed')
print('[INFO] Searching for NVIDIA GPU...')

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('[STATUS] NVIDIA GPU found, Training using GPU:', torch.cuda.get_device_name(0))
    model = model.cuda()
else:
    print('[STATUS] Nvidia GPU not found, training using CPU')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(args['learning_rate']))

print('[INFO] Training and evaluating the model...')

final_model, training_losses = train(int(args['epochs']), model, train_loader, criterion, optimizer, use_cuda, save_path='')
evaluate(model, test_loader, criterion, use_cuda)

print('[STATUS] Model training and evaluation completed')
print('Job completed!')