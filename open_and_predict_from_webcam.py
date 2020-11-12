import numpy as np
import argparse
import torch
import cv2

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-d', '--detector', required=True, help='Path to face detector file')
arg_parser.add_argument('-r', '--recognition_model', required=True, help='Path to face recognition file')
arg_parser.add_argument('-m', '--model', required=True, help='Path to the saved model')
args = vars(arg_parser.parse_args())
