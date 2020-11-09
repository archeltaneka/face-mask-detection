from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from torchvision import transforms, models

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

print(args['path'])
print(args['detector'])
print(args['recognition_model'])
print(args['resize'])
print(args['split_value'])
print(args['batch_size'])
print(args['epochs'])
print(args['learning_rate'])

