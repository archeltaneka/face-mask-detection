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
arg_parser.add_argument('-d', '-detector', required=True, help='Path to face detector file')
arg_parser.add_argument('-e', '--embedding-model', required=True, help='Path to face embedding file')
arg_parser.add_argument('-r', '--resize', required=True, help='Resize face to a certain size (must be the same size with the one you specified in extract_dataset.py)')
arg_parser.add_argument('-s', '--split-value', required=True, help='Train/test data split value (decimal point, e.g. 0.8)')
arg_parser.add_argument('-b', '--batch-size', required=True, help='Train/test batch size')
arg_parser.add_argument('-e', '--epochs', required=True, help='Number of training epochs')
arg_parser.add_argument('-l', '--learning-rate', required=True, help='Training learning rate')
args = vars(arg_parser.parse_args())

